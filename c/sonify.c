/*
 * sonify.c -- real-time PCA sonification core, C port of Todd's 2025 dissertation, Appendix C.3.
 * Reference implementation: sonify.py (sign_mode='aligned', svd_method='gram').  Interface: sonify.h.
 *
 * Data flow of one chunk (defaults: M=8 records x N=1500 bins):
 *   iq[5][M*N] --fftwf (5*M transforms, length N)--> PSD --@coeff--> B[4][M][N]
 *   per channel: row-mean subtract -> G = B B^T (MxM) -> Jacobi eigen -> keep >energy_pct
 *   -> newPower[N] = v^T B  (v = sign-aligned sum of the kept eigenvectors)
 *   -> tuner shift + conjugate-symmetric 2N spectrum -> c2r IFFT -> audio[2N][4]
 *
 * Real-time rules: everything is allocated in sonify_create(); sonify_process_chunk() does not allocate,
 * lock or print, and has no data-dependent loops except the bounded Jacobi sweeps.
 */
#include "sonify.h"

#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define JACOBI_MAX_SWEEPS 32

struct sonify_ctx {
    sonify_params p;
    sonify_dims   d;
    float         coeff[SONIFY_NANT][SONIFY_NBF];
    float         scale_psd;            /* 1 / (fs_rf * n)                     */
    float         scale_out;            /* fs_audio / (2 n)                    */

    fftwf_plan    plan_fwd;             /* in place, batch 5*M, length N       */
    fftwf_plan    plan_inv;             /* c2r, batch 4, length 2N             */
    fftwf_complex *cbuf;                /* [5*M*N]                             */
    float         *bp;                  /* [4][M][N]  B-Format power           */
    fftwf_complex *spec;                /* [4][N+1]   half spectra             */
    float         *tbuf;                /* [4][2N]    time-domain output       */
    float         *newp;                /* [N]                                 */
};

const char *sonify_status_str(sonify_status s)
{
    switch (s) {
    case SONIFY_OK:      return "ok";
    case SONIFY_E_PARAM: return "invalid parameters";
    case SONIFY_E_ALLOC: return "out of memory";
    case SONIFY_E_FFT:   return "FFTW could not create a plan";
    case SONIFY_E_NULL:  return "NULL pointer argument";
    }
    return "unknown status";
}

void sonify_params_default(sonify_params *p)
{
    p->fs_rf      = 120000.0;
    p->fs_audio   = 30000.0;
    p->t_chunk    = 0.1;
    p->tuner_hz   = -4000.0;
    p->energy_pct = 80.0;
    p->n_pca      = 4;
}

static int is_int(double x) { return fabs(x - floor(x + 0.5)) < 1e-6; }

sonify_status sonify_get_dims(const sonify_params *p, sonify_dims *d)
{
    if (!p || !d) return SONIFY_E_NULL;
    if (!(p->fs_rf > 0) || !(p->fs_audio > 0) || !(p->t_chunk > 0)) return SONIFY_E_PARAM;
    double m = 2.0 * p->fs_rf / p->fs_audio, n = p->t_chunk * p->fs_audio / 2.0, L = p->t_chunk * p->fs_rf;
    if (!is_int(m) || !is_int(n) || !is_int(L)) return SONIFY_E_PARAM;
    d->m         = (int)floor(m + 0.5);
    d->n         = (int)floor(n + 0.5);
    d->chunk_len = (int)floor(L + 0.5);
    d->out_len   = 2 * d->n;
    d->noffset   = (int)floor(fabs(p->tuner_hz) * p->t_chunk + 0.5);
    if (d->m < 1 || d->m > SONIFY_MAX_M || d->n < 2 || (d->n & 1)) return SONIFY_E_PARAM;
    if ((long)d->m * d->n != d->chunk_len) return SONIFY_E_PARAM;
    if (p->n_pca != 3 && p->n_pca != 4) return SONIFY_E_PARAM;
    if (!(p->energy_pct > 0.0 && p->energy_pct < 100.0)) return SONIFY_E_PARAM;
    if (p->tuner_hz < 0 && !(d->noffset >= 2 && d->noffset <= d->n)) return SONIFY_E_PARAM;
    if (p->tuner_hz > 0 && !(d->noffset <= d->n - 2)) return SONIFY_E_PARAM;
    return SONIFY_OK;
}

/* ------------------------------------------------------------------ diagnostics helpers */
float sonify_aoa_rf_rad(const sonify_chunk_info *i)  { return atan2f(i->rf_max[2],  i->rf_max[1]); }
float sonify_aoa_son_rad(const sonify_chunk_info *i) { return atan2f(i->son_max[2], i->son_max[1]); }
float sonify_el_rf_rad(const sonify_chunk_info *i)
{ return atan2f(i->rf_max[3], hypotf(i->rf_max[1], i->rf_max[2])); }
float sonify_el_son_rad(const sonify_chunk_info *i)
{ return atan2f(i->son_max[3], hypotf(i->son_max[1], i->son_max[2])); }

/* ------------------------------------------------------------------ create / destroy */
void sonify_destroy(sonify_ctx *c)
{
    if (!c) return;
    if (c->plan_fwd) fftwf_destroy_plan(c->plan_fwd);
    if (c->plan_inv) fftwf_destroy_plan(c->plan_inv);
    fftwf_free(c->cbuf);
    fftwf_free(c->bp);
    fftwf_free(c->spec);
    fftwf_free(c->tbuf);
    free(c->newp);
    free(c);
}

sonify_status sonify_create(sonify_ctx **out, const sonify_params *p,
                            const float coeff[SONIFY_NANT][SONIFY_NBF])
{
    if (!out) return SONIFY_E_NULL;
    *out = NULL;
    if (!p || !coeff) return SONIFY_E_NULL;

    sonify_dims d;
    sonify_status st = sonify_get_dims(p, &d);
    if (st != SONIFY_OK) return st;

    sonify_ctx *c = (sonify_ctx *)calloc(1, sizeof *c);
    if (!c) return SONIFY_E_ALLOC;
    c->p = *p;
    c->d = d;
    memcpy(c->coeff, coeff, sizeof c->coeff);
    c->scale_psd = (float)(1.0 / (p->fs_rf * d.n));
    c->scale_out = (float)(p->fs_audio / (2.0 * d.n));

    const size_t M = (size_t)d.m, N = (size_t)d.n;
    c->cbuf  = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * SONIFY_NANT * M * N);
    c->bp    = (float *)fftwf_malloc(sizeof(float) * SONIFY_NBF * M * N);
    c->spec  = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * SONIFY_NBF * (N + 1));
    c->tbuf  = (float *)fftwf_malloc(sizeof(float) * SONIFY_NBF * 2 * N);
    c->newp  = (float *)malloc(sizeof(float) * N);
    if (!c->cbuf || !c->bp || !c->spec || !c->tbuf || !c->newp) { sonify_destroy(c); return SONIFY_E_ALLOC; }

    int n_fwd = d.n;
    c->plan_fwd = fftwf_plan_many_dft(1, &n_fwd, SONIFY_NANT * d.m, c->cbuf, NULL, 1, d.n,
                                      c->cbuf, NULL, 1, d.n, FFTW_FORWARD, FFTW_MEASURE | FFTW_NO_BUFFERING);
    int n_inv = 2 * d.n;
    c->plan_inv = fftwf_plan_many_dft_c2r(1, &n_inv, SONIFY_NBF, c->spec, NULL, 1, d.n + 1,
                                          c->tbuf, NULL, 1, 2 * d.n, FFTW_MEASURE | FFTW_NO_BUFFERING);
    if (!c->plan_fwd || !c->plan_inv) { sonify_destroy(c); return SONIFY_E_FFT; }

    *out = c;
    return SONIFY_OK;
}

sonify_status sonify_set_coeff(sonify_ctx *c, const float coeff[SONIFY_NANT][SONIFY_NBF])
{
    if (!c || !coeff) return SONIFY_E_NULL;
    memcpy(c->coeff, coeff, sizeof c->coeff);
    return SONIFY_OK;
}

const sonify_dims *sonify_ctx_dims(const sonify_ctx *c) { return c ? &c->d : NULL; }

/* ------------------------------------------------------------------ stages */

/* chunk_psd + to_bformat_power: cbuf (after FFT) -> bp[b][m][n].
 * fftshift for even N: shifted[n] = raw[(n + N/2) % N]. */
static void bformat_power(sonify_ctx *c)
{
    const int M = c->d.m, N = c->d.n, h = N / 2;
    const float sc = c->scale_psd;
    float s[SONIFY_NANT];
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            const int j = (n < h) ? n + h : n - h;
            for (int a = 0; a < SONIFY_NANT; a++) {
                const fftwf_complex *x = &c->cbuf[((size_t)a * M + m) * N + j];
                s[a] = (x[0][0] * x[0][0] + x[0][1] * x[0][1]) * sc;
            }
            for (int b = 0; b < SONIFY_NBF; b++) {
                float acc = 0.f;
                for (int a = 0; a < SONIFY_NANT; a++) acc += s[a] * c->coeff[a][b];
                c->bp[((size_t)b * M + m) * N + n] = acc;
            }
        }
    }
}

/* signed_peak: per record the value at max |.| (first maximum, sign kept), mean over records. */
static void signed_peak(const sonify_ctx *c, float out[SONIFY_NBF])
{
    const int M = c->d.m, N = c->d.n;
    for (int b = 0; b < SONIFY_NBF; b++) {
        double acc = 0.0;
        for (int m = 0; m < M; m++) {
            const float *r = &c->bp[((size_t)b * M + m) * N];
            float best = r[0], bm = fabsf(r[0]);
            for (int n = 1; n < N; n++) {
                const float a = fabsf(r[n]);
                if (a > bm) { bm = a; best = r[n]; }
            }
            acc += best;
        }
        out[b] = (float)(acc / M);
    }
}

/* Cyclic Jacobi for a symmetric MxM matrix A (destroyed).  Eigenvalues -> w, eigenvectors -> columns of V.
 * Bounded sweeps; double precision. */
static void jacobi_eigen_sym(int M, double A[SONIFY_MAX_M][SONIFY_MAX_M],
                             double w[SONIFY_MAX_M], double V[SONIFY_MAX_M][SONIFY_MAX_M])
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) V[i][j] = (i == j);
    double fro = 0.0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) fro += A[i][j] * A[i][j];
    const double tol = 1e-15 * fro;                              /* on the sum of squares of off-diagonals */

    for (int sweep = 0; sweep < JACOBI_MAX_SWEEPS; sweep++) {
        double off = 0.0;
        for (int p = 0; p < M - 1; p++)
            for (int q = p + 1; q < M; q++) off += A[p][q] * A[p][q];
        if (off <= tol) break;
        for (int p = 0; p < M - 1; p++) {
            for (int q = p + 1; q < M; q++) {
                const double apq = A[p][q];
                if (apq == 0.0) continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                const double cs = 1.0 / sqrt(t * t + 1.0), sn = t * cs;
                for (int k = 0; k < M; k++) {                    /* A <- A J (columns p,q) */
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = cs * akp - sn * akq;
                    A[k][q] = sn * akp + cs * akq;
                }
                for (int k = 0; k < M; k++) {                    /* A <- J^T A (rows p,q)  */
                    const double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = cs * apk - sn * aqk;
                    A[q][k] = sn * apk + cs * aqk;
                }
                for (int k = 0; k < M; k++) {                    /* V <- V J */
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = cs * vkp - sn * vkq;
                    V[k][q] = sn * vkp + cs * vkq;
                }
            }
        }
    }
    for (int i = 0; i < M; i++) w[i] = A[i][i];
}

/* pca_channel: B (MxN, mean-subtracted in place) -> newp[N]; returns k (components kept). */
static int pca_channel(float *B, int M, int N, double energy_pct, float *newp)
{
    for (int m = 0; m < M; m++) {                                /* row-mean subtract (mean over N) */
        double s = 0.0;
        for (int n = 0; n < N; n++) s += B[(size_t)m * N + n];
        const float mu = (float)(s / N);
        for (int n = 0; n < N; n++) B[(size_t)m * N + n] -= mu;
    }

    double G[SONIFY_MAX_M][SONIFY_MAX_M], V[SONIFY_MAX_M][SONIFY_MAX_M], w[SONIFY_MAX_M];
    for (int i = 0; i < M; i++) {
        for (int j = i; j < M; j++) {
            double s = 0.0;
            const float *ri = &B[(size_t)i * N], *rj = &B[(size_t)j * N];
            for (int n = 0; n < N; n++) s += (double)ri[n] * rj[n];
            G[i][j] = G[j][i] = s;
        }
    }
    jacobi_eigen_sym(M, G, w, V);

    int order[SONIFY_MAX_M];                                     /* descending eigenvalue order */
    for (int i = 0; i < M; i++) { order[i] = i; if (!(w[i] > 0.0)) w[i] = 0.0; }
    for (int i = 1; i < M; i++) {
        const int o = order[i];
        int j = i - 1;
        while (j >= 0 && w[order[j]] < w[o]) { order[j + 1] = order[j]; j--; }
        order[j + 1] = o;
    }

    double tot = 0.0;
    for (int i = 0; i < M; i++) tot += w[i];
    if (!isfinite(tot) || tot <= 0.0) { memset(newp, 0, sizeof(float) * N); return 0; }

    int k = M;                                                   /* first component whose cumulative energy EXCEEDS pct */
    double cum = 0.0;
    for (int i = 0; i < M; i++) {
        cum += w[order[i]];
        if (100.0 * cum / tot > energy_pct) { k = i + 1; break; }
    }

    double v[SONIFY_MAX_M] = {0};                                /* v = sum_j s_j U[:,j],  s_j = sign(sum_m U[m][j]) (>= 0 -> +) */
    for (int j = 0; j < k; j++) {
        const int col = order[j];
        double c = 0.0;
        for (int m = 0; m < M; m++) c += V[m][col];
        const double s = (c < 0.0) ? -1.0 : 1.0;
        for (int m = 0; m < M; m++) v[m] += s * V[m][col];
    }
    for (int n = 0; n < N; n++) {
        double acc = 0.0;
        for (int m = 0; m < M; m++) acc += v[m] * B[(size_t)m * N + n];
        newp[n] = (float)acc;
    }
    return k;
}

/* place_spectrum: first half acs[0..N] of the 2N-point conjugate-symmetric spectrum (real, even);
 * the c2r transform supplies the mirrored half.  Same three tuner branches as C.3. */
static void place_half_spectrum(const float *newp, int N, int no, double tuner, fftwf_complex *spec)
{
    for (int k = 0; k <= N; k++) { spec[k][0] = 0.f; spec[k][1] = 0.f; }
    if (tuner == 0.0) {
        for (int k = 1; k < N; k++) spec[k][0] = newp[k];
    } else if (tuner > 0.0) {                                    /* acs = [0, zeros(no), newp[1:N-no], 0, ...] */
        for (int i = 0; i < N - no - 1; i++) spec[1 + no + i][0] = newp[1 + i];
    } else {                                                     /* acs = [0, newp[no-1:], zeros(no-2), 0, ...] */
        for (int i = 0; i < N - no + 1; i++) spec[1 + i][0] = newp[no - 1 + i];
    }
}

/* ------------------------------------------------------------------ one chunk */
sonify_status sonify_process_chunk(sonify_ctx *c, const float *iq, float *audio_out, sonify_chunk_info *info)
{
    if (!c || !iq || !audio_out) return SONIFY_E_NULL;
    const int M = c->d.m, N = c->d.n, L = c->d.chunk_len, T2 = 2 * N;

    memcpy(c->cbuf, iq, sizeof(float) * 2 * (size_t)SONIFY_NANT * L);
    fftwf_execute(c->plan_fwd);
    bformat_power(c);

    sonify_chunk_info loc;
    sonify_chunk_info *inf = info ? info : &loc;
    memset(inf, 0, sizeof *inf);
    signed_peak(c, inf->rf_max);

    for (int b = 0; b < SONIFY_NBF; b++) {
        fftwf_complex *sp = &c->spec[(size_t)b * (N + 1)];
        if (b < c->p.n_pca) {
            const int k = pca_channel(&c->bp[(size_t)b * M * N], M, N, c->p.energy_pct, c->newp);
            inf->kept[b] = k;
            if (k) {
                int im = 0;
                for (int n = 1; n < N; n++) if (fabsf(c->newp[n]) > fabsf(c->newp[im])) im = n;
                inf->son_max[b] = c->newp[im];
            }
            place_half_spectrum(c->newp, N, c->d.noffset, c->p.tuner_hz, sp);
        } else {
            for (int k = 0; k <= N; k++) { sp[k][0] = 0.f; sp[k][1] = 0.f; }   /* Z off (n_pca == 3) */
        }
    }
    fftwf_execute(c->plan_inv);

    for (int b = 0; b < SONIFY_NBF; b++) {
        const float *t = &c->tbuf[(size_t)b * T2];
        for (int n = 0; n < T2; n++) audio_out[(size_t)n * SONIFY_NBF + b] = t[n] * c->scale_out;
    }
    return SONIFY_OK;
}

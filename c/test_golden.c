/*
 * test_golden.c -- validates sonify.c against the Python reference (golden/<case>/...) and checks
 * the extra properties listed in sonify.h.  Also times process_chunk.
 *
 *   ./test_golden [golden_dir]          (default ./golden)
 *
 * Linux/glibc: malloc/free/calloc/realloc/posix_memalign are interposed to prove that
 * sonify_process_chunk() performs no heap activity.  On macOS that check is skipped.
 */
#define _GNU_SOURCE
#include "sonify.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__GLIBC__)
#  define COUNT_ALLOCS 1
extern void *__libc_malloc(size_t), *__libc_calloc(size_t, size_t), *__libc_realloc(void *, size_t),
            *__libc_memalign(size_t, size_t);
extern void  __libc_free(void *);
static volatile long g_allocs;
void *malloc(size_t n)                   { g_allocs++; return __libc_malloc(n); }
void *calloc(size_t a, size_t b)         { g_allocs++; return __libc_calloc(a, b); }
void *realloc(void *p, size_t n)         { g_allocs++; return __libc_realloc(p, n); }
void  free(void *p)                      { __libc_free(p); }
int   posix_memalign(void **r, size_t al, size_t n)
{ g_allocs++; *r = __libc_memalign(al, n); return *r ? 0 : 12; }
void *aligned_alloc(size_t al, size_t n) { g_allocs++; return __libc_memalign(al, n); }
void *memalign(size_t al, size_t n)      { g_allocs++; return __libc_memalign(al, n); }
#else
#  define COUNT_ALLOCS 0
static volatile long g_allocs;
#endif

static int g_fail;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("    FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static void *read_file(const char *dir, const char *name, size_t *bytes)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("    cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[n] = 0;                                   /* text files (params.json) are parsed as strings */
    if (bytes) *bytes = (size_t)n;
    return buf;
}

/* minimal params.json reader: finds  "key": number */
static int json_num(const char *txt, const char *key, double *out)
{
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(txt, pat);
    if (!p) return 0;
    *out = atof(p + strlen(pat));
    return 1;
}

static double max_abs(const float *x, size_t n) { double m = 0; for (size_t i = 0; i < n; i++) if (fabs(x[i]) > m) m = fabs(x[i]); return m; }
static double max_diff(const float *a, const float *b, size_t n) { double m = 0; for (size_t i = 0; i < n; i++) if (fabs((double)a[i] - b[i]) > m) m = fabs((double)a[i] - b[i]); return m; }

static void test_case(const char *root, const char *name)
{
    char dir[1024];
    snprintf(dir, sizeof dir, "%s/%s", root, name);
    printf("[%s]\n", name);

    char *js = (char *)read_file(dir, "params.json", NULL);
    float *iq = (float *)read_file(dir, "iq.f32", NULL), *coeff = (float *)read_file(dir, "coeff.f32", NULL);
    float *ref = (float *)read_file(dir, "audio.f32", NULL), *rinfo = (float *)read_file(dir, "info.f32", NULL);
    int32_t *rkept = (int32_t *)read_file(dir, "kept.i32", NULL);
    if (!js || !iq || !coeff || !ref || !rinfo || !rkept) { g_fail++; return; }

    sonify_params p;
    sonify_params_default(&p);
    double v;
    if (json_num(js, "fsU", &v)) p.fs_rf = v;
    if (json_num(js, "fsA", &v)) p.fs_audio = v;
    if (json_num(js, "Tchunk", &v)) p.t_chunk = v;
    if (json_num(js, "tuner", &v)) p.tuner_hz = v;
    if (json_num(js, "energy_pct", &v)) p.energy_pct = v;
    if (json_num(js, "n_pca", &v)) p.n_pca = (int)v;

    sonify_ctx *c;
    sonify_status st = sonify_create(&c, &p, (const float (*)[SONIFY_NBF])coeff);
    CHECK(st == SONIFY_OK, "sonify_create: %s", sonify_status_str(st));
    if (st != SONIFY_OK) return;
    const sonify_dims *d = sonify_ctx_dims(c);
    const size_t na = (size_t)d->out_len * SONIFY_NBF;
    float *out = (float *)calloc(na, sizeof(float)), *out2 = (float *)calloc(na, sizeof(float));
    sonify_chunk_info info, info2;

    long a0 = g_allocs;
    st = sonify_process_chunk(c, iq, out, &info);
    CHECK(st == SONIFY_OK, "process_chunk: %s", sonify_status_str(st));

    const double ea = max_diff(out, ref, na) / max_abs(ref, na);
    const double ei = max_diff((float[8]){info.rf_max[0], info.rf_max[1], info.rf_max[2], info.rf_max[3],
                                          info.son_max[0], info.son_max[1], info.son_max[2], info.son_max[3]},
                               rinfo, 8) / max_abs(rinfo, 8);
    printf("    kept C=(%d,%d,%d,%d) ref=(%d,%d,%d,%d)   rel.err audio %.1e  info %.1e\n",
           info.kept[0], info.kept[1], info.kept[2], info.kept[3], rkept[0], rkept[1], rkept[2], rkept[3], ea, ei);
    for (int b = 0; b < 4; b++) CHECK(info.kept[b] == rkept[b], "kept[%d] %d != %d", b, info.kept[b], rkept[b]);
    CHECK(ea <= 1e-3, "audio rel.err %.2e > 1e-3", ea);
    CHECK(ei <= 1e-3, "info rel.err %.2e > 1e-3", ei);

    /* determinism / chunk independence: same input twice -> identical output */
    sonify_process_chunk(c, iq, out2, &info2);
    CHECK(memcmp(out, out2, na * sizeof(float)) == 0, "second call on the same input differs");
    CHECK(memcmp(&info, &info2, sizeof info) == 0, "second call info differs");

    /* AoA / elevation helpers vs reference peaks */
    printf("    az RF %.2f deg, sonified %.2f deg | el RF %.2f deg, sonified %.2f deg\n",
           sonify_aoa_rf_rad(&info) * 57.29578, sonify_aoa_son_rad(&info) * 57.29578,
           sonify_el_rf_rad(&info) * 57.29578, sonify_el_son_rad(&info) * 57.29578);

    /* timing + no heap activity inside process_chunk */
    const int reps = 300;
    double tmin = 1e9, tsum = 0, tmax = 0;
    for (int i = 0; i < reps; i++) {
        double t0 = now_s();
        sonify_process_chunk(c, iq, out2, NULL);
        double dt = now_s() - t0;
        if (dt < tmin) tmin = dt;
        if (dt > tmax) tmax = dt;
        tsum += dt;
    }
    long allocs = g_allocs - a0;
    printf("    process_chunk: min %.3f ms  mean %.3f ms  max %.3f ms  (chunk = %.0f ms of data, mean real-time factor %.0fx)\n",
           tmin * 1e3, tsum / reps * 1e3, tmax * 1e3, p.t_chunk * 1e3, p.t_chunk / (tsum / reps));
    if (COUNT_ALLOCS) CHECK(allocs == 0, "%ld heap allocations inside process_chunk", allocs);
    else printf("    (heap-activity check skipped on this platform)\n");

    /* n_pca == 3 : Z output exactly 0, W X Y unchanged vs n_pca == 4 */
    if (p.n_pca == 4) {
        sonify_params p3 = p; p3.n_pca = 3;
        sonify_ctx *c3;
        if (sonify_create(&c3, &p3, (const float (*)[SONIFY_NBF])coeff) == SONIFY_OK) {
            float *o3 = (float *)calloc(na, sizeof(float));
            sonify_chunk_info i3;
            sonify_process_chunk(c3, iq, o3, &i3);
            double zmax = 0, dmax = 0;
            for (int n = 0; n < d->out_len; n++) {
                zmax = fmax(zmax, fabs(o3[n * 4 + 3]));
                for (int b = 0; b < 3; b++) dmax = fmax(dmax, fabs((double)o3[n * 4 + b] - out[n * 4 + b]));
            }
            CHECK(zmax == 0.0, "n_pca=3: Z not exactly 0 (%.2e)", zmax);
            CHECK(dmax == 0.0, "n_pca=3: W,X,Y differ from n_pca=4 (%.2e)", dmax);
            CHECK(i3.kept[3] == 0 && i3.son_max[3] == 0.f, "n_pca=3: Z info not zero");
            printf("    n_pca=3 vs 4: Z == 0 and W,X,Y identical: %s\n", (zmax == 0.0 && dmax == 0.0) ? "yes" : "NO");
            free(o3);
            sonify_destroy(c3);
        }
    }

    /* conjugate symmetry => real, even-symmetric spectrum round trip: DFT of the audio must have zero DC and Nyquist */
    {
        double dc[4] = {0}, nyq[4] = {0}, pk[4] = {0};
        for (int n = 0; n < d->out_len; n++)
            for (int b = 0; b < 4; b++) {
                const double x = out[n * 4 + b];
                dc[b] += x; nyq[b] += (n & 1) ? -x : x; pk[b] = fmax(pk[b], fabs(x));
            }
        double worst = 0;
        for (int b = 0; b < 4; b++) if (pk[b] > 0) worst = fmax(worst, fmax(fabs(dc[b]), fabs(nyq[b])) / (pk[b] * d->out_len));
        CHECK(worst < 1e-5, "spectrum DC/Nyquist bins not zero (%.2e)", worst);
    }

    free(js); free(iq); free(coeff); free(ref); free(rinfo); free(rkept); free(out); free(out2);
    sonify_destroy(c);
}

static void test_params(void)
{
    printf("[parameter validation]\n");
    sonify_params p; sonify_dims d;
    sonify_params_default(&p);
    CHECK(sonify_get_dims(&p, &d) == SONIFY_OK && d.m == 8 && d.n == 1500 && d.chunk_len == 12000 &&
          d.out_len == 3000 && d.noffset == 400, "default dims");
    sonify_params q = p; q.n_pca = 5;                 CHECK(sonify_get_dims(&q, &d) == SONIFY_E_PARAM, "n_pca 5 must fail");
    q = p; q.fs_audio = 29000;                        CHECK(sonify_get_dims(&q, &d) == SONIFY_E_PARAM, "non-integer M must fail");
    q = p; q.tuner_hz = -10;                          CHECK(sonify_get_dims(&q, &d) == SONIFY_E_PARAM, "noffset < 2 must fail");
    q = p; q.tuner_hz = 15000;                        CHECK(sonify_get_dims(&q, &d) == SONIFY_E_PARAM, "positive tuner too large must fail");
    q = p; q.energy_pct = 100;                        CHECK(sonify_get_dims(&q, &d) == SONIFY_E_PARAM, "energy 100 must fail");
    float cf[5][4] = {{0}};
    sonify_ctx *c = (void *)1;
    CHECK(sonify_create(&c, NULL, cf) == SONIFY_E_NULL && c == NULL, "NULL params");
    CHECK(sonify_process_chunk(NULL, NULL, NULL, NULL) == SONIFY_E_NULL, "NULL ctx");
    printf("    done\n");
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : "golden";
    test_params();
    const char *cases[] = {"snr60_tunerM4000", "snr40_tunerM4000", "snr60_tunerP4000", "snr60_tuner0"};
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) test_case(root, cases[i]);
    printf(g_fail ? "\n%d CHECK(S) FAILED\n" : "\nALL C CHECKS PASSED\n", g_fail);
    return g_fail != 0;
}

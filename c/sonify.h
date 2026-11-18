/*
 * sonify.h -- real-time PCA sonification core, C port of Todd's 2025 dissertation, Appendix C.3 "RFAA Full System".
 *
 * Python reference + golden test vectors:  sonify.py, make_golden.py, golden/<case>/...
 *
 * One call processes ONE chunk (default 0.1 s):
 *
 *   iq[5 antennas][chunk_len][I,Q]  --FFT/PSD-->  Sxx[M][N][5]  --@coeff-->  B[M][N][W,X,Y,Z]
 *        --per channel W,X,Y (+Z if n_pca==4): mean-subtract, Gram eigen-decomposition, keep >energy_pct,
 *          sum, sign-align-->
 *        newPower[N]  --tuner shift + conjugate-symmetric 2N spectrum--> c2r IFFT --> audio[out_len][W,X,Y,Z]
 *
 * 3-D: with n_pca == 4 the Z channel is sonified too (full-sphere sources, elevation available through
 * sonify_el_*_rad()); n_pca == 3 is Todd's horizontal-plane system (Z output exactly 0).  The 5x4
 * coefficient matrix must be fitted for the chosen mode (Z column non-zero for 3-D).
 *
 * Python name           ->  C (static, in sonify.c)
 *   chunk_psd           ->  chunk_psd()          one batched fftwf plan: 5*M complex transforms of length N
 *   to_bformat_power    ->  bformat_power()      (M*N x 5) * (5 x 4), float
 *   signed_peak         ->  signed_peak()        argmax |.| per record, keep sign, mean over records
 *   pca_channel         ->  pca_channel()        B*B^T (MxM, double) -> jacobi_eigen_sym() -> Z = U^T B
 *   place_spectrum      ->  place_spectrum()     three tuner branches (<0, 0, >0), identical to C.3
 *   np.fft.ifft*fsA     ->  fftwf c2r plan (2N), scaled by fs_audio/(2N)
 *
 * Design rules (real-time):
 *   - ALL memory and FFTW plans are created in sonify_create().  sonify_process_chunk() never calls
 *     malloc/free/printf, takes no locks, and its run time is data independent except for the number
 *     of Jacobi sweeps (bounded).
 *   - A context is NOT re-entrant: use one context per thread.  Contexts are independent.
 *   - Only the deterministic 'aligned' sign mode is implemented.  Todd's literal signB recipe depends on
 *     the sign convention of the SVD routine (see sonify.py) and exists in Python only.
 *
 * Build (macOS / Apple Silicon, Homebrew):
 *   cc -std=gnu11 -O2 -Wall -Wextra -I/opt/homebrew/include sonify.c test_golden.c \
 *      -L/opt/homebrew/lib -lfftw3f -lm -o test_golden      (or just: make)
 * Build (Ubuntu):  sudo apt install libfftw3-dev   then the same command without the -I/-L paths.
 */
#ifndef SONIFY_H
#define SONIFY_H

#ifdef __cplusplus
extern "C" {
#endif

#define SONIFY_NANT   5    /* RF antennas                                           */
#define SONIFY_NBF    4    /* B-Format channels, always in the order W X Y Z        */
#define SONIFY_NPCA   4    /* max sonified channels: W X Y Z (params.n_pca selects 3 or 4) */
#define SONIFY_MAX_M  16   /* largest supported records-per-chunk (static solver buffers) */

typedef enum {
    SONIFY_OK      =  0,
    SONIFY_E_PARAM = -1,   /* invalid parameters, see sonify_get_dims()             */
    SONIFY_E_ALLOC = -2,   /* out of memory in sonify_create()                      */
    SONIFY_E_FFT   = -3,   /* FFTW could not create a plan                          */
    SONIFY_E_NULL  = -4    /* NULL pointer argument                                 */
} sonify_status;

const char *sonify_status_str(sonify_status s);

/* ------------------------------------------------------------------ parameters */
typedef struct {
    double fs_rf;        /* fsU:   RF baseband sample rate [Hz]          default 120000 */
    double fs_audio;     /* fsA:   audio sample rate [Hz]                default  30000 */
    double t_chunk;      /* Tchunk: sonification window [s]              default    0.1 */
    double tuner_hz;     /* frequency shift [Hz]; <0 down, >0 up, 0 off  default  -4000 */
    double energy_pct;   /* keep PCA components until this % of energy is EXCEEDED, default 80 */
    int    n_pca;        /* 3 = sonify W,X,Y (Z output = 0, Todd);  4 = also Z (3-D)   default 4 */
} sonify_params;

void sonify_params_default(sonify_params *p);

/* Derived sizes.  With the defaults: m=8, n=1500, chunk_len=12000, out_len=3000, noffset=400. */
typedef struct {
    int m;           /* records per chunk            = 2*fs_rf/fs_audio                    */
    int n;           /* samples (= PSD bins) / record = t_chunk*fs_audio/2  (must be even)  */
    int chunk_len;   /* complex RF samples per antenna per chunk = m*n                      */
    int out_len;     /* audio frames per chunk       = 2*n                                  */
    int noffset;     /* |tuner| in audio-spectrum bins, = round(|tuner_hz|*t_chunk)         */
} sonify_dims;

/* Checks that m*n == t_chunk*fs_rf, m is an integer <= SONIFY_MAX_M, n is even, n_pca is 3 or 4,
 * tuner<0 -> 2 <= noffset <= n, tuner>0 -> noffset <= n-2.  Returns SONIFY_OK or SONIFY_E_PARAM. */
sonify_status sonify_get_dims(const sonify_params *p, sonify_dims *d);

/* ------------------------------------------------------------------ per-chunk diagnostics */
typedef struct {
    float rf_max[SONIFY_NBF];    /* RFMaxVal: record-averaged SIGNED peak of B-Format power, before sonification */
    float son_max[SONIFY_NBF];   /* maxVal:   signed peak of the reconstructed spectrum after PCA (Z = 0 if n_pca == 3) */
    int   kept[SONIFY_NPCA];     /* principal components kept for W, X, Y, Z (Z entry 0 if n_pca == 3)            */
} sonify_chunk_info;

/* Azimuth = atan2(Y, X) in radians, range (-pi, pi].  Per chunk; average with a CIRCULAR mean, not a plain mean. */
float sonify_aoa_rf_rad(const sonify_chunk_info *i);    /* from rf_max  (before sonification) */
float sonify_aoa_son_rad(const sonify_chunk_info *i);   /* from son_max (after sonification)  */

/* Elevation = atan2(Z, hypot(X, Y)) in radians, range [-pi/2, pi/2]; 0 when n_pca == 3 (Z = 0).
 * Azimuth is undefined at the poles (X = Y = 0): for errors use the great-circle angle between unit vectors
 * (X,Y,Z)/|(X,Y,Z)|, see sonify.py doa_vec / great_circle_deg / mean_direction. */
float sonify_el_rf_rad(const sonify_chunk_info *i);
float sonify_el_son_rad(const sonify_chunk_info *i);

/* ------------------------------------------------------------------ context */
typedef struct sonify_ctx sonify_ctx;     /* opaque */

/* coeff: 5x4 beamforming matrix, row = antenna, column = W,X,Y,Z (same layout as numpy (5,4) float32).
 * It is copied.  *out is set to NULL on failure. */
sonify_status sonify_create(sonify_ctx **out, const sonify_params *p,
                            const float coeff[SONIFY_NANT][SONIFY_NBF]);
void          sonify_destroy(sonify_ctx *c);                 /* accepts NULL */

/* Replace the coefficient matrix (e.g. after re-calibration); no allocation. */
sonify_status sonify_set_coeff(sonify_ctx *c, const float coeff[SONIFY_NANT][SONIFY_NBF]);

/* Sizes the context was built for (valid as long as the context lives). */
const sonify_dims *sonify_ctx_dims(const sonify_ctx *c);

/* ------------------------------------------------------------------ processing */
/*
 * iq        : chunk_len complex samples for each of the 5 antennas, antenna-major, I/Q interleaved float32:
 *             iq[(a*chunk_len + s)*2 + 0] = I,  iq[(a*chunk_len + s)*2 + 1] = Q.
 *             (== numpy complex64 array of shape (5, chunk_len); == 5 consecutive fread()s of Todd's
 *              Ch0Data..Ch4Data raw float32 files.)
 * audio_out : out_len interleaved frames [W,X,Y,Z] of float32, i.e. out_len*4 floats (numpy (out_len,4)).
 *             The level is NOT normalised (same as the thesis); Z is exactly 0 only if n_pca == 3.
 * info      : optional (may be NULL).
 * iq and audio_out must not overlap.
 */
sonify_status sonify_process_chunk(sonify_ctx *c, const float *iq, float *audio_out,
                                   sonify_chunk_info *info);

/*
 * ------------------------------------------------------------------ test contract (golden/<case>/)
 *   params.json  -> sonify_params (incl. n_pca) + dims    iq.f32    -> input  (5, chunk_len, 2) float32
 *   coeff.f32    -> (5,4) float32                         audio.f32 -> expected (out_len, 4) float32
 *   info.f32     -> rf_max[4], son_max[4] (8 float32)     kept.i32  -> expected kept[4] (int32)
 * The current golden set is 3-D (source az 60 deg, el +35 deg, n_pca = 4); the horizontal n_pca = 3 set
 * of the first version is in v1_horizontal/golden (its kept.i32 has 3 entries).
 * Pass criteria for sonify_process_chunk():
 *   kept            : exactly equal
 *   audio, info     : max|c - ref| / max|ref| <= 1e-3   (a float32 pipeline should reach ~1e-5..1e-4;
 *                                                          the float32 INPUT rounding alone is 1e-7)
 * Golden chunks are chosen >= 1 percentage point away from the energy threshold in the noisy case,
 * so float32 vs float64 cannot legitimately disagree on `kept`.
 * Extra checks to add in C: chunk-to-chunk independence (same input twice -> same output), Z == 0 for n_pca == 3,
 * conjugate symmetry of the spectrum (round trip), no heap activity inside process_chunk (e.g. run under
 * valgrind --tool=massif on Ubuntu, or malloc-hook on macOS).
 *
 * ------------------------------------------------------------------ minimal usage
 *   sonify_params p; sonify_params_default(&p);
 *   sonify_ctx *c; float coeff[5][4] = { ... };
 *   if (sonify_create(&c, &p, coeff) != SONIFY_OK) { ... }
 *   const sonify_dims *d = sonify_ctx_dims(c);
 *   float *iq  = malloc(sizeof(float) * 2 * SONIFY_NANT * d->chunk_len);
 *   float *aud = malloc(sizeof(float) * 4 * d->out_len);
 *   for (;;) { fill iq; sonify_process_chunk(c, iq, aud, &info); play/write aud; }
 *   sonify_destroy(c);
 */

#ifdef __cplusplus
}
#endif
#endif /* SONIFY_H */

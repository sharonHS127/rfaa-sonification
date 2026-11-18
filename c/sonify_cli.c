/*
 * sonify_cli.c -- stream front-end for sonify.c.  Reads 5-antenna RF baseband IQ from a file OR a pipe /
 * stdin as it arrives (live), sonifies it chunk by chunk and writes audio to a WAV file or raw to stdout.
 *
 *   sonify_cli --coeff coeff.f32 -i rf.f32 -o out.wav                     # recorded / simulated file
 *   sonify_cli --coeff coeff.f32 -i - --channels 2 -o - | aplay -f S16_LE -c 2 -r 30000   # live, Linux
 *   <capture program> | sonify_cli --coeff coeff.f32 -i - --layout interleaved -o - | ...   # live pipe
 *
 * Input layouts (per chunk of chunk_len complex samples per antenna, I/Q interleaved):
 *   --layout chunk        antenna-major:   [ant0: s0..sL-1][ant1: ...]...[ant4: ...]   (= golden iq.f32, Todd's Ch0..Ch4 files)
 *   --layout interleaved  time-major:      s0:[ant0 I Q, ant1 I Q, ... ant4 I Q], s1:[...] ...
 * Input sample formats: --format f32 (default) or s16 (scaled by 1/32768).
 */
#define _POSIX_C_SOURCE 200809L
#include "sonify.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void sleep_until(double t)
{
    double dt = t - now_s();
    if (dt <= 0) return;
    struct timespec ts = { (time_t)dt, (long)((dt - floor(dt)) * 1e9) };
    nanosleep(&ts, NULL);
}

static void usage(void)
{
    fputs(
"usage: sonify_cli --coeff FILE [options]\n"
"  --coeff FILE      5x4 float32 matrix (rows = antennas, cols = W X Y Z), raw little endian   (required)\n"
"  -i FILE|-         input (default -, stdin)             -o FILE|-   output; FILE.wav = WAV, '-' = raw stdout\n"
"  --layout chunk|interleaved      input layout (default chunk)\n"
"  --format f32|s16                input sample format (default f32)\n"
"  --channels 4|2|1                output: 4 = W X Y Z, 2 = quick stereo preview (L=W+Y, R=W-Y), 1 = W (default 4)\n"
"  --out-format s16|f32            raw stdout sample format (default s16; WAV files are always s16)\n"
"  --gain G|auto                   output gain; auto (default) = 0.5/peak of the FIRST chunk, then held fixed\n"
"  --tuner HZ  --fs-rf HZ  --fs-audio HZ  --tchunk S  --energy PCT  --npca 3|4\n"
"  --pace            sleep so that chunks are consumed at real-time speed (makes a file behave like a live feed)\n"
"  --log FILE        per-chunk CSV: t, az/el RF, az/el sonified, kept, process ms\n"
"  --quiet           no statistics on stderr\n", stderr);
}

typedef struct { FILE *f; int nch; uint32_t rate; uint64_t frames; } wavw;

static void put32(FILE *f, uint32_t v) { fputc(v & 255, f); fputc(v >> 8 & 255, f); fputc(v >> 16 & 255, f); fputc(v >> 24 & 255, f); }
static void put16(FILE *f, uint16_t v) { fputc(v & 255, f); fputc(v >> 8 & 255, f); }
static void wav_header(wavw *w)
{
    const uint32_t bytes = (uint32_t)(w->frames * w->nch * 2);
    fseek(w->f, 0, SEEK_SET);
    fwrite("RIFF", 1, 4, w->f); put32(w->f, 36 + bytes); fwrite("WAVEfmt ", 1, 8, w->f);
    put32(w->f, 16); put16(w->f, 1); put16(w->f, (uint16_t)w->nch); put32(w->f, w->rate);
    put32(w->f, w->rate * w->nch * 2); put16(w->f, (uint16_t)(w->nch * 2)); put16(w->f, 16);
    fwrite("data", 1, 4, w->f); put32(w->f, bytes);
}

/* read exactly n bytes from a pipe/file; returns bytes read (< n only at EOF) */
static size_t read_full(FILE *f, void *buf, size_t n)
{
    size_t got = 0;
    while (got < n) {
        size_t r = fread((char *)buf + got, 1, n - got, f);
        if (r == 0) break;
        got += r;
    }
    return got;
}

int main(int argc, char **argv)
{
    sonify_params p;
    sonify_params_default(&p);
    const char *coeff_path = NULL, *in_path = "-", *out_path = NULL, *log_path = NULL;
    int interleaved = 0, s16_in = 0, nch_out = 4, s16_out = 1, pace = 0, quiet = 0, auto_gain = 1;
    double gain = 1.0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT() (i + 1 < argc ? argv[++i] : (usage(), exit(2), (char *)0))
        if (!strcmp(a, "--coeff")) coeff_path = NEXT();
        else if (!strcmp(a, "-i")) in_path = NEXT();
        else if (!strcmp(a, "-o")) out_path = NEXT();
        else if (!strcmp(a, "--layout")) interleaved = !strcmp(NEXT(), "interleaved");
        else if (!strcmp(a, "--format")) s16_in = !strcmp(NEXT(), "s16");
        else if (!strcmp(a, "--channels")) nch_out = atoi(NEXT());
        else if (!strcmp(a, "--out-format")) s16_out = strcmp(NEXT(), "f32") != 0;
        else if (!strcmp(a, "--gain")) { const char *g = NEXT(); if (!strcmp(g, "auto")) auto_gain = 1; else { auto_gain = 0; gain = atof(g); } }
        else if (!strcmp(a, "--tuner")) p.tuner_hz = atof(NEXT());
        else if (!strcmp(a, "--fs-rf")) p.fs_rf = atof(NEXT());
        else if (!strcmp(a, "--fs-audio")) p.fs_audio = atof(NEXT());
        else if (!strcmp(a, "--tchunk")) p.t_chunk = atof(NEXT());
        else if (!strcmp(a, "--energy")) p.energy_pct = atof(NEXT());
        else if (!strcmp(a, "--npca")) p.n_pca = atoi(NEXT());
        else if (!strcmp(a, "--pace")) pace = 1;
        else if (!strcmp(a, "--log")) log_path = NEXT();
        else if (!strcmp(a, "--quiet")) quiet = 1;
        else { usage(); return 2; }
    }
    if (!coeff_path || !out_path || (nch_out != 1 && nch_out != 2 && nch_out != 4)) { usage(); return 2; }

    float coeff[SONIFY_NANT][SONIFY_NBF];
    FILE *fc = fopen(coeff_path, "rb");
    if (!fc || fread(coeff, sizeof(float), 20, fc) != 20) { fprintf(stderr, "cannot read 5x4 float32 matrix from %s\n", coeff_path); return 1; }
    fclose(fc);

    sonify_ctx *ctx;
    sonify_status st = sonify_create(&ctx, &p, (const float (*)[SONIFY_NBF])coeff);
    if (st != SONIFY_OK) { fprintf(stderr, "sonify_create: %s\n", sonify_status_str(st)); return 1; }
    const sonify_dims *d = sonify_ctx_dims(ctx);
    const size_t L = (size_t)d->chunk_len, per_chunk = (size_t)SONIFY_NANT * L * 2;     /* floats (or s16 values) per chunk */

    FILE *fin = strcmp(in_path, "-") ? fopen(in_path, "rb") : stdin;
    if (!fin) { perror(in_path); return 1; }
    int to_stdout = !strcmp(out_path, "-");
    wavw w = { NULL, nch_out, (uint32_t)(p.fs_audio + 0.5), 0 };
    if (!to_stdout) { w.f = fopen(out_path, "wb+"); if (!w.f) { perror(out_path); return 1; } wav_header(&w); }
    FILE *flog = log_path ? fopen(log_path, "w") : NULL;
    if (flog) fprintf(flog, "t_s,az_rf_deg,el_rf_deg,az_son_deg,el_son_deg,kept_W,kept_X,kept_Y,kept_Z,proc_ms\n");

    void  *raw   = malloc(per_chunk * (s16_in ? sizeof(int16_t) : sizeof(float)));
    float *iq    = (float *)malloc(per_chunk * sizeof(float));
    float *audio = (float *)malloc(sizeof(float) * SONIFY_NBF * d->out_len);
    int16_t *pcm = (int16_t *)malloc(sizeof(int16_t) * nch_out * d->out_len);
    float   *fout = (float *)malloc(sizeof(float) * nch_out * d->out_len);
    if (!raw || !iq || !audio || !pcm || !fout) { fputs("out of memory\n", stderr); return 1; }

    const double chunk_s = p.t_chunk;
    double t_start = now_s(), tsum = 0, tmax = 0, wsum = 0;
    long nchunk = 0, missed = 0, clipped = 0;
    double peak_seen = 0;

    for (;;) {
        const size_t esz = s16_in ? sizeof(int16_t) : sizeof(float);
        if (read_full(fin, raw, per_chunk * esz) != per_chunk * esz) break;           /* EOF / partial chunk dropped */
        if (pace)                                   /* chunk k is complete at t_start + (k+1)*chunk_s in a live system */
            sleep_until(t_start + (double)(nchunk + 1) * chunk_s);

        const double t_in = now_s();                                                   /* chunk fully received */
        /* -> antenna-major float32 */
        if (!interleaved) {
            for (size_t k = 0; k < per_chunk; k++) iq[k] = s16_in ? ((const int16_t *)raw)[k] / 32768.f : ((const float *)raw)[k];
        } else {
            for (size_t s = 0; s < L; s++)
                for (int a = 0; a < SONIFY_NANT; a++)
                    for (int q = 0; q < 2; q++) {
                        const size_t src = (s * SONIFY_NANT + a) * 2 + q, dst = ((size_t)a * L + s) * 2 + q;
                        iq[dst] = s16_in ? ((const int16_t *)raw)[src] / 32768.f : ((const float *)raw)[src];
                    }
        }

        sonify_chunk_info info;
        const double t0 = now_s();
        st = sonify_process_chunk(ctx, iq, audio, &info);
        const double dt = now_s() - t0;
        if (st != SONIFY_OK) { fprintf(stderr, "process_chunk: %s\n", sonify_status_str(st)); return 1; }

        /* channel selection, level */
        for (int n = 0; n < d->out_len; n++) {
            const float *f = &audio[(size_t)n * 4];
            if (nch_out == 4) { for (int b = 0; b < 4; b++) fout[n * 4 + b] = f[b]; }
            else if (nch_out == 2) { fout[n * 2] = f[0] + f[2]; fout[n * 2 + 1] = f[0] - f[2]; }
            else fout[n] = f[0];
        }
        const size_t ns = (size_t)nch_out * d->out_len;
        if (auto_gain && nchunk == 0) {
            double pk = 0; for (size_t k = 0; k < ns; k++) pk = fmax(pk, fabs(fout[k]));
            gain = pk > 0 ? 0.5 / pk : 1.0;
            if (!quiet) fprintf(stderr, "auto gain %.4g (first chunk peak %.4g)\n", gain, pk);
        }
        for (size_t k = 0; k < ns; k++) { fout[k] *= (float)gain; peak_seen = fmax(peak_seen, fabs(fout[k])); }

        if (to_stdout && !s16_out) fwrite(fout, sizeof(float), ns, stdout);
        else {
            for (size_t k = 0; k < ns; k++) {
                float v = fout[k] * 32767.f;
                if (v > 32767.f) { v = 32767.f; clipped++; } else if (v < -32768.f) { v = -32768.f; clipped++; }
                pcm[k] = (int16_t)lrintf(v);
            }
            fwrite(pcm, sizeof(int16_t), ns, to_stdout ? stdout : w.f);
        }
        if (to_stdout) fflush(stdout);
        w.frames += (uint64_t)d->out_len;
        const double lat = now_s() - t_in;                                             /* receive -> audio written */

        if (flog) fprintf(flog, "%.3f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%.3f\n", nchunk * chunk_s,
                          sonify_aoa_rf_rad(&info) * 57.29578, sonify_el_rf_rad(&info) * 57.29578,
                          sonify_aoa_son_rad(&info) * 57.29578, sonify_el_son_rad(&info) * 57.29578,
                          info.kept[0], info.kept[1], info.kept[2], info.kept[3], dt * 1e3);
        tsum += dt; wsum += lat; if (dt > tmax) tmax = dt;
        if (dt > chunk_s) missed++;
        nchunk++;
    }

    if (!to_stdout) { wav_header(&w); fclose(w.f); }
    if (flog) fclose(flog);
    if (!quiet && nchunk > 0)
        fprintf(stderr, "%ld chunks (%.1f s of RF) | process %.3f ms mean, %.3f ms max (budget %.0f ms, deadline misses %ld) | "
                        "receive->audio %.3f ms mean | output peak %.2f, clipped samples %ld\n",
                nchunk, nchunk * chunk_s, tsum / nchunk * 1e3, tmax * 1e3, chunk_s * 1e3, missed, wsum / nchunk * 1e3, peak_seen, clipped);
    if (!quiet && nchunk == 0) fputs("no complete chunk was read\n", stderr);
    sonify_destroy(ctx);
    free(raw); free(iq); free(audio); free(pcm); free(fout);
    if (fin != stdin) fclose(fin);
    return 0;
}

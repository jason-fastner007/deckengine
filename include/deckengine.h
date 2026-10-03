/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>
 *
 * deckengine — C API
 *
 * Low-latency two-deck audio engine for Linux and Windows.
 *
 * Conventions
 *   - Functions returning int32_t return DE_OK (0) or a negative DE_ERR_* code.
 *     de_last_error() then describes the failure (valid until the next call on that thread).
 *   - Strings are NUL-terminated UTF-8.
 *   - Decks: 0 = A, 1 = B.
 *   - All functions are thread-safe and return quickly; none blocks on the audio thread.
 *     Exceptions: de_engine_start / de_engine_free open/close the device, de_deck_load opens
 *     and probes the file, de_analyze_file decodes a whole file.
 */
#ifndef DECKENGINE_H
#define DECKENGINE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DE_OK              0
#define DE_ERR_NULL       (-1)
#define DE_ERR_IO         (-2)
#define DE_ERR_DECODE     (-3)
#define DE_ERR_DEVICE     (-4)
#define DE_ERR_NO_TRACK   (-5)
#define DE_ERR_QUEUE_FULL (-6)
#define DE_ERR_INVALID    (-7)
#define DE_ERR_PANIC      (-8)

#define DE_DECK_A 0u
#define DE_DECK_B 1u

#define DE_CURVE_EQUAL_POWER 0u
#define DE_CURVE_LINEAR      1u
#define DE_CURVE_ADDITIVE    2u

typedef struct DeEngine DeEngine;

/* Output settings. Zero-initialise for defaults. */
typedef struct DeOutputConfig {
    uint32_t    buffer_frames;        /* frames per callback, 0 = 128 */
    uint32_t    sample_rate;          /* 0 = device default */
    const char *device;               /* device id or name substring, NULL = default */
    const char *host;                 /* "alsa", "pipewire", "jack", "wasapi", "asio", ... NULL = auto */
    float       limiter_lookahead_ms; /* 0 = 1 ms */
} DeOutputConfig;

typedef struct DeDeckStatus {
    bool     loaded;
    bool     playing;
    bool     ended;
    bool     failed;
    double   position_seconds;
    double   duration_seconds;   /* < 0 when unknown */
    float    speed;
    float    peak;
    uint64_t underruns;
} DeDeckStatus;

typedef struct DeStats {
    uint32_t sample_rate;
    uint32_t buffer_frames;
    uint32_t max_buffer_frames;
    uint64_t callbacks;
    double   process_avg_us;
    double   process_max_us;
    double   dsp_load;           /* 0..1 */
    uint64_t deadline_misses;
    uint64_t backend_errors;
    double   buffer_ms;
    double   device_latency_ms;
    double   engine_latency_ms;
    double   output_latency_ms;
    float    peak_l;
    float    peak_r;
    float    short_term_lufs;
    float    leveler_gain_db;
    float    limiter_gain_db;
    float    crossfader;
} DeStats;

typedef struct DeLoudness {
    float integrated_lufs;
    float true_peak_db;
    float loudness_range_lu;
} DeLoudness;

/* ── Library ─────────────────────────────────────────────────────────────── */
const char *de_version(void);
const char *de_last_error(void);

/* Blocking: decodes the whole file. Run on a worker thread. */
int32_t de_analyze_file(const char *path, DeLoudness *out);

/* ── Engine ──────────────────────────────────────────────────────────────── */
DeEngine *de_engine_start(const DeOutputConfig *config); /* NULL on failure */
void      de_engine_free(DeEngine *engine);
uint32_t  de_engine_sample_rate(const DeEngine *engine);
uint64_t  de_engine_clock(const DeEngine *engine);        /* frames rendered so far */
int32_t   de_engine_stats(const DeEngine *engine, DeStats *out);

/* ── Decks ───────────────────────────────────────────────────────────────── */
int32_t de_deck_load(const DeEngine *engine, uint32_t deck, const char *path);
int32_t de_deck_load_pcm(const DeEngine *engine, uint32_t deck, const float *interleaved_stereo,
                         uint64_t frames, uint32_t sample_rate);
int32_t de_deck_unload(const DeEngine *engine, uint32_t deck);
int32_t de_deck_play(const DeEngine *engine, uint32_t deck);
int32_t de_deck_play_at(const DeEngine *engine, uint32_t deck, uint64_t at_frame);
int32_t de_deck_pause(const DeEngine *engine, uint32_t deck);
int32_t de_deck_seek(const DeEngine *engine, uint32_t deck, double seconds);
int32_t de_deck_set_loop(const DeEngine *engine, uint32_t deck, double start_s, double end_s);
int32_t de_deck_clear_loop(const DeEngine *engine, uint32_t deck);
int32_t de_deck_set_speed(const DeEngine *engine, uint32_t deck, float ratio);
int32_t de_deck_set_volume(const DeEngine *engine, uint32_t deck, float volume);
int32_t de_deck_set_trim_db(const DeEngine *engine, uint32_t deck, float db);
/* bass/drums: 0 kill · 1 normal · 2 boost. vocal: 0 instrumental · 1 normal · 2 a cappella */
int32_t de_deck_set_stems(const DeEngine *engine, uint32_t deck, float bass, float vocal, float drums);
int32_t de_deck_status(const DeEngine *engine, uint32_t deck, DeDeckStatus *out);

/* ── Mixer / master ──────────────────────────────────────────────────────── */
int32_t de_set_crossfader(const DeEngine *engine, float position);               /* 0 = A, 1 = B */
int32_t de_crossfade(const DeEngine *engine, float position, double seconds, uint64_t at_frame); /* at_frame 0 = now */
int32_t de_set_crossfade_curve(const DeEngine *engine, uint32_t curve);
int32_t de_set_master_volume(const DeEngine *engine, float volume);
int32_t de_set_normalization(const DeEngine *engine, bool enabled, float target_lufs);
int32_t de_set_limiter(const DeEngine *engine, bool enabled);

#ifdef __cplusplus
}
#endif

#endif /* DECKENGINE_H */

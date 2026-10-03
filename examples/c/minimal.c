/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>
 *
 * Minimal C client. Build (Linux):
 *   cargo build --release
 *   cc examples/c/minimal.c -Iinclude -Ltarget/release -ldeckengine -o minimal
 *   LD_LIBRARY_PATH=target/release ./minimal song.mp3
 */
#define _DEFAULT_SOURCE /* usleep */
#include <stdio.h>
#include <deckengine.h>

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <unistd.h>
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

int main(int argc, char **argv) {
    printf("deckengine %s\n", de_version());

    DeOutputConfig cfg = {0};
    cfg.buffer_frames = 128;
    DeEngine *e = de_engine_start(&cfg);
    if (!e) {
        fprintf(stderr, "start failed: %s\n", de_last_error());
        return 1;
    }

    if (argc > 1) {
        if (de_deck_load(e, DE_DECK_A, argv[1]) != DE_OK) {
            fprintf(stderr, "load failed: %s\n", de_last_error());
            de_engine_free(e);
            return 1;
        }
    } else {
        /* One second of silence, just to exercise the API. */
        static float silence[48000 * 2];
        de_deck_load_pcm(e, DE_DECK_A, silence, 48000, 48000);
    }
    de_deck_play(e, DE_DECK_A);

    for (int i = 0; i < 3; i++) {
        sleep_ms(1000);
        DeStats s;
        DeDeckStatus d;
        de_engine_stats(e, &s);
        de_deck_status(e, DE_DECK_A, &d);
        printf("pos %.2fs  buffer %u frames  output latency %.2f ms  dsp %.1f%%\n",
               d.position_seconds, s.buffer_frames, s.output_latency_ms, s.dsp_load * 100.0);
    }

    /* Error path check. */
    if (de_deck_seek(e, 7, 1.0) != DE_ERR_INVALID) {
        fprintf(stderr, "expected DE_ERR_INVALID\n");
        return 1;
    }
    printf("error message works: %s\n", de_last_error());

    de_engine_free(e);
    return 0;
}

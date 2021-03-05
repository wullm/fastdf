/*  Adapted from the code included on Sebastian Vigna's website */

#include <stdint.h>

#define XOR_RAND_MAX UINT64_MAX

static inline uint64_t rol64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

struct xoshiro256ss_state {
    uint64_t s[4];
};

static inline uint64_t xoshiro256ss(struct xoshiro256ss_state *state) {
	uint64_t *s = state->s;
	uint64_t const result = rol64(s[1] * 5, 7) * 9;
	uint64_t const t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;
	s[3] = rol64(s[3], 45);

	return result;
}


/* A second random number generator, used to seed the first */

struct splitmix64_state {
    uint64_t s;
};

static inline uint64_t splitmix64(struct splitmix64_state *state) {
    uint64_t result = state->s;

    state->s = result + 0x9E3779B97f4A7C15;
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

/* Finally, an initialization method for the xoshiro rng using splitmix64 rng */

static inline struct xoshiro256ss_state xoshiro256ss_init(uint64_t seed) {
    struct splitmix64_state smstate = {seed};
    struct xoshiro256ss_state result = {0};

    result.s[0] = splitmix64(&smstate);
    result.s[1] = splitmix64(&smstate);
    result.s[2] = splitmix64(&smstate);
    result.s[3] = splitmix64(&smstate);

    return result;
};

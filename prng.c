#include "./include/prng.h"

/*  xoshiro256++ 1.0 and splitmix64
 *
 *  Written in 2018–2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 *
 *  To the extent possible under law, the authors have dedicated all copyright
 *  and related and neighboring rights to this software to the public domain
 *  worldwide. This software is distributed without any warranty.
 *
 *  You should have received a copy of the CC0 Public Domain Dedication along
 *  with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 */

/* ---------- xoshiro256++ state and core ---------- */
u64 s[4];

u64 rotl(const u64 x, int k) {
    return (x << k) | (x >> (64 - k));
}

// xoshiro256++ 1.0: returns a 64-bit random number.
u64 next(void) {
    const u64 result = rotl(s[0] + s[3], 23) + s[0];

    const u64 t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);

    return result;
}

f64 randn(void) {
    // uniform (0,1) from 53 high bits of next()
    f64 u1 = (next() >> 11) * 0x1.0p-53;
    f64 u2 = (next() >> 11) * 0x1.0p-53;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ---------- splitmix64 seeder ---------- */

u64 splitmix_state;

uint64_t splitmix64(void) {
    uint64_t z = (splitmix_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Initialize xoshiro256++ state from a single 64-bit seed using splitmix64
void rng_seed(uint64_t seed) {
    splitmix_state = seed;
    s[0] = splitmix64();
    s[1] = splitmix64();
    s[2] = splitmix64();
    s[3] = splitmix64();
}

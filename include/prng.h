#ifndef PRNG_H
#define PRNG_H

#include "utility.h"

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

// xoshiro256++ state and core
extern u64 s[4];
extern u64 splitmix_state;

u64 rotl(const u64 x, int k);
u64 next(void);
void rng_seed(u64 seed);
f64 randn(void);

#endif

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <xsimd/xsimd.hpp>

#include "random/macros.hpp"
#include "xsimd/config/xsimd_arch.hpp"
#include "xsimd/types/xsimd_api.hpp"

namespace prng {

namespace internal {
  template <std::uint8_t R>
  struct ChaChaSIMDCreator;
}

template <std::uint8_t R = 20>
class ChaChaSIMD {
public:
  static constexpr auto MATRIX_ROW_LEN = std::uint8_t{4};
  static constexpr auto MATRIX_COL_LEN = std::uint8_t{4};
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto CACHE_BLOCKCOUNT = std::uint16_t{std::numeric_limits<std::uint8_t>::max() + 1};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;
  using cache_type = std::array<matrix_word, CACHE_BLOCKCOUNT * MATRIX_WORDCOUNT>;
  using rounds_type = std::uint8_t;

  /**
   * @brief Construct a SIMD ChaCha generator with given key, counter and nonce
   * @param key A 256-bit key, divided up into eight 32-bit words.
   * @param counter Initial value of the counter.
   * @param nonce Initial value of the nonce.
   */
  explicit PRNG_ALWAYS_INLINE ChaChaSIMD(
    const std::array<matrix_word, KEY_WORDCOUNT> key,
    const input_word counter,
    const input_word nonce
  ) {

    // TODO: Test with more than the default arch
    pImpl = xsimd::dispatch<xsimd::arch_list<xsimd::default_arch>>(internal::ChaChaSIMDCreator<R>{m_state, m_cache})();

    // First four words (i.e. top-row) are always the same constants
    // They spell out "expand 2-byte k" in ASCII (little-endian)
    m_state[0] = 0x61707865;
    m_state[1] = 0x3320646e;
    m_state[2] = 0x79622d32;
    m_state[3] = 0x6b206574;

    for (auto i = 0; i < 8; ++i) {
      m_state[4 + i] = key[i];
    }

    // ChaCha assumes little-endianness
    m_state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
    m_state[13] = static_cast<matrix_word>(counter >> 32);
    m_state[14] = static_cast<matrix_word>(nonce & 0xFFFFFFFF);
    m_state[15] = static_cast<matrix_word>(nonce >> 32);
  }

  /**
   * @brief Generates the next random block.
   * @return The next random block.
   */
  PRNG_ALWAYS_INLINE constexpr matrix_type operator()() noexcept {
    if (m_index == 0) [[unlikely]] {
      pImpl->populate_cache();
    }
    matrix_word *cache_block = m_cache.data() + (m_index++ * MATRIX_WORDCOUNT);
    matrix_type out_block;
    std::copy(cache_block, cache_block + MATRIX_WORDCOUNT, out_block.begin());
    return out_block;
  }

  /**
   * Abstract interface to hide the templated implementation.
   */
  struct IChaChaSIMD {
    virtual ~IChaChaSIMD() = default;
    virtual void populate_cache() = 0;
  };

private:
  matrix_type m_state;
  std::unique_ptr<IChaChaSIMD> pImpl;
  cache_type m_cache;
  std::uint8_t m_index{0}; // Make sure this is 0
};



namespace internal {

template <class Arch, std::uint8_t R>
class ChaChaSIMDImpl : public ChaChaSIMD<>::IChaChaSIMD {
public:
  using matrix_word = ChaChaSIMD<>::matrix_word;
  using matrix_type = ChaChaSIMD<>::matrix_type;
  using cache_type =  ChaChaSIMD<>::cache_type;
  using simd_type = xsimd::batch<matrix_word, Arch>;

  static constexpr std::uint8_t SIMD_WIDTH = std::uint8_t{simd_type::size};
  static constexpr auto MATRIX_WORDCOUNT = ChaChaSIMD<>::MATRIX_WORDCOUNT;
  static constexpr auto CACHE_BLOCKCOUNT = std::uint16_t{std::numeric_limits<std::uint8_t>::max() + 1};
  static constexpr auto CACHE_WORDCOUNT = std::uint16_t{CACHE_BLOCKCOUNT * MATRIX_WORDCOUNT};

  PRNG_ALWAYS_INLINE explicit ChaChaSIMDImpl(matrix_type &state, cache_type &cache) : m_state{state}, m_cache{cache} {};

  PRNG_ALWAYS_INLINE void populate_cache() noexcept override {
    std::cout << Arch::name() << std::endl;
    for (auto i = 0; i < CACHE_WORDCOUNT; i += SIMD_WIDTH * MATRIX_WORDCOUNT) {
      next_blocks(m_cache.data() + i);
    }
  }

private:
  matrix_type &m_state;
  cache_type &m_cache;

  /**
   * Return an array { 0, 1 * step, ..., (n - 1) * step }. Can be used to initialize a batch for
   * consecutively incremeting elements in a batch of low counter words, as well as
   * a batch of offsets to scatter matrix words into memory with.
  */
  template <size_t... Is>
  static constexpr PRNG_ALWAYS_INLINE std::array<matrix_word, sizeof...(Is)> matrix_word_sequence(std::index_sequence<Is...>, std::uint8_t step = 1) noexcept {
    return {static_cast<matrix_word>(Is * step)...};
  }

  /**
   * Return an array { 0, n < 1, n < 2, ..., n < (SIMD_WIDTH - 1) }. Can be used to initialize a
   * batch for incremetinng elements in a batch of consecutive high counter words, depending on at
   * what index the corresponding lower counter words had an overflow.
   */
  // TODO: Restore the macro PRNG_ALWAYS_INLINE use here, I removed it cuz it messes up treesitter.
  static constexpr std::array<matrix_word, SIMD_WIDTH> make_higher_counter_inc(std::uint8_t n) noexcept {
    std::array<matrix_word, SIMD_WIDTH> incs;
    incs[0] = 0;
    for (auto i = 1; i < SIMD_WIDTH; ++i) {
      incs[i] = n < i;
    }
    return incs;
  }

  /**
   * Given an initial state, generates `SIMD_WIDTH` new ChaCha blocks and
   * stores them sequentially at the given address. Will increment the state's
   * counter words by `SIMD_WIDTH`.
   */
  PRNG_ALWAYS_INLINE constexpr void next_blocks(matrix_word *out) noexcept {
    simd_type lower_counter_inc, higher_counter_inc;
    lower_counter_inc =
      xsimd::load_unaligned(matrix_word_sequence(std::make_index_sequence<SIMD_WIDTH>{}).data());
    matrix_word overflow_index = std::numeric_limits<matrix_word>::max() - m_state[12];
    if (overflow_index < SIMD_WIDTH) [[unlikely]] {
      higher_counter_inc = xsimd::load_unaligned(make_higher_counter_inc(overflow_index).data());
    } else {
      higher_counter_inc = simd_type::broadcast(0);
    }


    simd_type x[MATRIX_WORDCOUNT];
    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i] = simd_type::broadcast(m_state[i]);
    }
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;

    for (auto i = 0; i < R; i += 2) {
      x[0] += x[4];
      x[1] += x[5];
      x[2] += x[6];
      x[3] += x[7];

      x[12] ^= x[0];
      x[13] ^= x[1];
      x[14] ^= x[2];
      x[15] ^= x[3];

      x[12] = xsimd::rotl<16>(x[12]);
      x[13] = xsimd::rotl<16>(x[13]);
      x[14] = xsimd::rotl<16>(x[14]);
      x[15] = xsimd::rotl<16>(x[15]);

      x[8] += x[12];
      x[9] += x[13];
      x[10] += x[14];
      x[11] += x[15];

      x[4] ^= x[8];
      x[5] ^= x[9];
      x[6] ^= x[10];
      x[7] ^= x[11];

      x[4] = xsimd::rotl<12>(x[4]);
      x[5] = xsimd::rotl<12>(x[5]);
      x[6] = xsimd::rotl<12>(x[6]);
      x[7] = xsimd::rotl<12>(x[7]);

      x[0] += x[4];
      x[1] += x[5];
      x[2] += x[6];
      x[3] += x[7];

      x[12] ^= x[0];
      x[13] ^= x[1];
      x[14] ^= x[2];
      x[15] ^= x[3];

      x[12] = xsimd::rotl<8>(x[12]);
      x[13] = xsimd::rotl<8>(x[13]);
      x[14] = xsimd::rotl<8>(x[14]);
      x[15] = xsimd::rotl<8>(x[15]);

      x[8] += x[12];
      x[9] += x[13];
      x[10] += x[14];
      x[11] += x[15];

      x[4] ^= x[8];
      x[5] ^= x[9];
      x[6] ^= x[10];
      x[7] ^= x[11];

      x[4] = xsimd::rotl<7>(x[4]);
      x[5] = xsimd::rotl<7>(x[5]);
      x[6] = xsimd::rotl<7>(x[6]);
      x[7] = xsimd::rotl<7>(x[7]);

      x[0] += x[5];
      x[1] += x[6];
      x[2] += x[7];
      x[3] += x[4];

      x[15] ^= x[0];
      x[12] ^= x[1];
      x[13] ^= x[2];
      x[14] ^= x[3];

      x[15] = xsimd::rotl<16>(x[15]);
      x[12] = xsimd::rotl<16>(x[12]);
      x[13] = xsimd::rotl<16>(x[13]);
      x[14] = xsimd::rotl<16>(x[14]);

      x[10] += x[15];
      x[11] += x[12];
      x[8] += x[13];
      x[9] += x[14];

      x[5] ^= x[10];
      x[6] ^= x[11];
      x[7] ^= x[8];
      x[4] ^= x[9];

      x[5] = xsimd::rotl<12>(x[5]);
      x[6] = xsimd::rotl<12>(x[6]);
      x[7] = xsimd::rotl<12>(x[7]);
      x[4] = xsimd::rotl<12>(x[4]);

      x[0] += x[5];
      x[1] += x[6];
      x[2] += x[7];
      x[3] += x[4];

      x[15] ^= x[0];
      x[12] ^= x[1];
      x[13] ^= x[2];
      x[14] ^= x[3];

      x[15] = xsimd::rotl<8>(x[15]);
      x[12] = xsimd::rotl<8>(x[12]);
      x[13] = xsimd::rotl<8>(x[13]);
      x[14] = xsimd::rotl<8>(x[14]);

      x[10] += x[15];
      x[11] += x[12];
      x[8] += x[13];
      x[9] += x[14];

      x[5] ^= x[10];
      x[6] ^= x[11];
      x[7] ^= x[8];
      x[4] ^= x[9];

      x[5] = xsimd::rotl<7>(x[5]);
      x[6] = xsimd::rotl<7>(x[6]);
      x[7] = xsimd::rotl<7>(x[7]);
      x[4] = xsimd::rotl<7>(x[4]);
    }

    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i] += simd_type::broadcast(m_state[i]);
    }
    // Remember to apply counter increments when summing rounds results with the original states.
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;

    // Batch i contains the i'th word of all chacha blocks, so transpose to get rows of chacha blocks.
    simd_type scatter_offsets =
      xsimd::load_unaligned(matrix_word_sequence(std::make_index_sequence<SIMD_WIDTH>{}, MATRIX_WORDCOUNT).data());
    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i].scatter(out + i, scatter_offsets);
    }

    m_state[12] += SIMD_WIDTH;
    m_state[13] += overflow_index < SIMD_WIDTH;
  }
};


template <std::uint8_t R>
struct ChaChaSIMDCreator {
  typename ChaChaSIMD<R>::matrix_type &state;
  typename ChaChaSIMD<R>::cache_type &cache;
  template <class Arch>
  std::unique_ptr<typename ChaChaSIMD<R>::IChaChaSIMD> operator()(Arch) const;
};

template <std::uint8_t R>
template <class Arch>
std::unique_ptr<typename ChaChaSIMD<R>::IChaChaSIMD> ChaChaSIMDCreator<R>::operator()(Arch) const {
  return std::make_unique<ChaChaSIMDImpl<Arch, R>>(state, cache);
};












} // namespace internal

} // namespace prng

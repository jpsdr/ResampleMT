// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://www.avisynth.org

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.

#include <stdio.h>
#include "resample.h"
#include "avs/config.h"
#include "avs/alignment.h"

#define VERSION "ResampleMT 1.0.0 JPSDR"

#define myCloseHandle(ptr) if (ptr!=NULL) { CloseHandle(ptr); ptr=NULL;}

// Intrinsics for SSE4.1, SSSE3, SSE3, SSE2, ISSE and MMX
#include <smmintrin.h>

/***************************************
 ********* Templated SSE Loader ********
 ***************************************/

typedef __m128i (SSELoader)(const __m128i*);

__forceinline __m128i simd_load_aligned(const __m128i* adr)
{
  return _mm_load_si128(adr);
}

__forceinline __m128i simd_load_unaligned(const __m128i* adr)
{
  return _mm_loadu_si128(adr);
}

__forceinline __m128i simd_load_unaligned_sse3(const __m128i* adr)
{
  return _mm_lddqu_si128(adr);
}

__forceinline __m128i simd_load_streaming(const __m128i* adr)
{
  return _mm_stream_load_si128(const_cast<__m128i*>(adr));
}

/***************************************
 ***** Vertical Resizer Assembly *******
 ***************************************/

template<typename pixel_size>
static void resize_v_planar_pointresize(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width,  int MinY, int MaxY, const int* pitch_table, const void* storage)
{
  int filter_size = program->filter_size;
  
  pixel_size* src0 = (pixel_size *)src;
  pixel_size* dst0 = (pixel_size *)dst;
  dst_pitch = dst_pitch / sizeof(pixel_size);

  for (int y = MinY; y < MaxY; y++) {
    int offset = program->pixel_offset[y];
	const pixel_size* src_ptr = src0 + pitch_table[offset];
    
	memcpy(dst0, src_ptr, width*sizeof(pixel_size));

    dst0 += dst_pitch;
  }
}


static void resize_v_c_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int MinY, int MaxY, const int* pitch_table, const void* storage)
{
  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;
  current_coeff+=filter_size*MinY;

  for (int y = MinY; y < MaxY; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + pitch_table[offset];

    for (int x = 0; x < width; x++) {
      int result = 0;
      for (int i = 0; i < filter_size; i++)
        result += (src_ptr+pitch_table[i])[x] * current_coeff[i];
	  result = (result+8192) >> 14;
      result = result > 255 ? 255 : result < 0 ? 0 : result;
      dst[x] = (BYTE) result;
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}


static void resize_v_c_planar_f(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int MinY, int MaxY, const int* pitch_table, const void* storage)
{
  int filter_size = program->filter_size;
  float* current_coeff = program->pixel_coefficient_float;

  current_coeff+=filter_size*MinY;

  float* src0 = (float *)src;
  float* dst0 = (float *)dst;
  dst_pitch = dst_pitch >> 2;

  for (int y = MinY; y < MaxY; y++)
  {
    int offset = program->pixel_offset[y];
	const float* src_ptr = src0 + pitch_table[offset];

    for (int x = 0; x < width; x++)
	{
      float result = 0;
      for (int i = 0; i < filter_size; i++)
		result += (src_ptr+pitch_table[i])[x] * current_coeff[i];
      dst0[x] = result;
    }

    dst0 += dst_pitch;
    current_coeff += filter_size;
  }
}


static void resize_v_c_planar_s(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int MinY, int MaxY, const int* pitch_table, const void* storage)
{
  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;

  current_coeff+=filter_size*MinY;

  uint16_t* src0 = (uint16_t *)src;
  uint16_t* dst0 = (uint16_t *)dst;
  dst_pitch = dst_pitch >> 1;

  for (int y = MinY; y < MaxY; y++)
  {
    int offset = program->pixel_offset[y];
	const uint16_t* src_ptr = src0 + pitch_table[offset];

    for (int x = 0; x < width; x++)
	{
      __int64 result = 0;
      for (int i = 0; i < filter_size; i++)
		result += (src_ptr+pitch_table[i])[x] * current_coeff[i];
	  result = (result+8192) >> 14;
      result = result > 65535 ? 65535 : result < 0 ? 0 : result;
      dst0[x] = (uint16_t) result;
    }

    dst0 += dst_pitch;
    current_coeff += filter_size;
  }
}


#ifdef X86_32
static void resize_v_mmx_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width,  int MinY, int MaxY, const int* pitch_table, const void* storage)
{
  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient + filter_size*MinY;

  int wMod8 = (width >> 3) << 3;
  int sizeMod2 = (filter_size >> 1) << 1;
  bool notMod2 = sizeMod2 < filter_size;

  __m64 zero = _mm_setzero_si64();

  for (int y = MinY; y < MaxY; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + pitch_table[offset];

    for (int x = 0; x < wMod8; x += 8) {
      __m64 result_1 = _mm_set1_pi32(8192); // Init. with rounder (16384/2 = 8192)
      __m64 result_2 = result_1;
      __m64 result_3 = result_1;
      __m64 result_4 = result_1;

      for (int i = 0; i < sizeMod2; i += 2) {
        __m64 src_p1 = *(reinterpret_cast<const __m64*>(src_ptr+pitch_table[i]+x));   // For detailed explanation please see SSE2 version.
        __m64 src_p2 = *(reinterpret_cast<const __m64*>(src_ptr+pitch_table[i+1]+x));

        __m64 src_l = _mm_unpacklo_pi8(src_p1, src_p2);                                   
        __m64 src_h = _mm_unpackhi_pi8(src_p1, src_p2);                                   

        __m64 src_1 = _mm_unpacklo_pi8(src_l, zero);                                      
        __m64 src_2 = _mm_unpackhi_pi8(src_l, zero);                                      
        __m64 src_3 = _mm_unpacklo_pi8(src_h, zero);                                      
        __m64 src_4 = _mm_unpackhi_pi8(src_h, zero);                                      

        __m64 coeff = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(current_coeff+i));   
        coeff = _mm_unpacklo_pi32(coeff, coeff);                                               

        __m64 dst_1 = _mm_madd_pi16(src_1, coeff);                                        
        __m64 dst_2 = _mm_madd_pi16(src_2, coeff);                                        
        __m64 dst_3 = _mm_madd_pi16(src_3, coeff);
        __m64 dst_4 = _mm_madd_pi16(src_4, coeff);

        result_1 = _mm_add_pi32(result_1, dst_1);
        result_2 = _mm_add_pi32(result_2, dst_2);
        result_3 = _mm_add_pi32(result_3, dst_3);
        result_4 = _mm_add_pi32(result_4, dst_4);
      }

      if (notMod2) { // do last odd row
        __m64 src_p = *(reinterpret_cast<const __m64*>(src_ptr+pitch_table[sizeMod2]+x));

        __m64 src_l = _mm_unpacklo_pi8(src_p, zero);
        __m64 src_h = _mm_unpackhi_pi8(src_p, zero);

        __m64 coeff = _mm_set1_pi16(current_coeff[sizeMod2]);

        __m64 dst_ll = _mm_mullo_pi16(src_l, coeff);   // Multiply by coefficient
        __m64 dst_lh = _mm_mulhi_pi16(src_l, coeff);
        __m64 dst_hl = _mm_mullo_pi16(src_h, coeff);
        __m64 dst_hh = _mm_mulhi_pi16(src_h, coeff);

        __m64 dst_1 = _mm_unpacklo_pi16(dst_ll, dst_lh); // Unpack to 32-bit integer
        __m64 dst_2 = _mm_unpackhi_pi16(dst_ll, dst_lh);
        __m64 dst_3 = _mm_unpacklo_pi16(dst_hl, dst_hh);
        __m64 dst_4 = _mm_unpackhi_pi16(dst_hl, dst_hh);

        result_1 = _mm_add_pi32(result_1, dst_1);
        result_2 = _mm_add_pi32(result_2, dst_2);
        result_3 = _mm_add_pi32(result_3, dst_3);
        result_4 = _mm_add_pi32(result_4, dst_4);
      }

      // Divide by 16348 (FPRound)
      result_1  = _mm_srai_pi32(result_1, 14);
      result_2  = _mm_srai_pi32(result_2, 14);
      result_3  = _mm_srai_pi32(result_3, 14);
      result_4  = _mm_srai_pi32(result_4, 14);

      // Pack and store
      __m64 result_l = _mm_packs_pi32(result_1, result_2);
      __m64 result_h = _mm_packs_pi32(result_3, result_4);
      __m64 result   = _mm_packs_pu16(result_l, result_h);

      *(reinterpret_cast<__m64*>(dst+x)) = result;
    }

    // Leftover
    for (int x = wMod8; x < width; x++)
	{
      int result = 0;

      for (int i = 0; i < filter_size; i++)
        result += (src_ptr+pitch_table[i])[x] * current_coeff[i];
	  result = (result+8192) >> 14;
      result = result > 255 ? 255 : result < 0 ? 0 : result;
      dst[x] = (BYTE) result;
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }

  _mm_empty();
}
#endif

template<SSELoader load>
static void resize_v_sse2_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width,  int MinY, int MaxY, const int* pitch_table, const void* storage)
{
  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient + filter_size*MinY;
  
  int wMod16 = (width >> 4) << 4;
  int sizeMod2 = (filter_size >> 1) << 1;
  bool notMod2 = sizeMod2 < filter_size;

  __m128i zero = _mm_setzero_si128();

  for (int y = MinY; y < MaxY; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + pitch_table[offset];

    for (int x = 0; x < wMod16; x += 16) {
      __m128i result_1 = _mm_set1_epi32(8192); // Init. with rounder (16384/2 = 8192)
      __m128i result_2 = result_1;
      __m128i result_3 = result_1;
      __m128i result_4 = result_1;
      
      for (int i = 0; i < sizeMod2; i += 2) {
        __m128i src_p1 = load(reinterpret_cast<const __m128i*>(src_ptr+pitch_table[i]+x));   // p|o|n|m|l|k|j|i|h|g|f|e|d|c|b|a
        __m128i src_p2 = load(reinterpret_cast<const __m128i*>(src_ptr+pitch_table[i+1]+x)); // P|O|N|M|L|K|J|I|H|G|F|E|D|C|B|A
         
        __m128i src_l = _mm_unpacklo_epi8(src_p1, src_p2);                                   // Hh|Gg|Ff|Ee|Dd|Cc|Bb|Aa
        __m128i src_h = _mm_unpackhi_epi8(src_p1, src_p2);                                   // Pp|Oo|Nn|Mm|Ll|Kk|Jj|Ii

        __m128i src_1 = _mm_unpacklo_epi8(src_l, zero);                                      // .D|.d|.C|.c|.B|.b|.A|.a
        __m128i src_2 = _mm_unpackhi_epi8(src_l, zero);                                      // .H|.h|.G|.g|.F|.f|.E|.e
        __m128i src_3 = _mm_unpacklo_epi8(src_h, zero);                                      // etc.
        __m128i src_4 = _mm_unpackhi_epi8(src_h, zero);                                      // etc.

        __m128i coeff = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(current_coeff+i));   // XX|XX|XX|XX|XX|XX|CO|co
        coeff = _mm_shuffle_epi32(coeff, 0);                                                 // CO|co|CO|co|CO|co|CO|co
        
        __m128i dst_1 = _mm_madd_epi16(src_1, coeff);                                         // CO*D+co*d | CO*C+co*c | CO*B+co*b | CO*A+co*a
        __m128i dst_2 = _mm_madd_epi16(src_2, coeff);                                         // etc.
        __m128i dst_3 = _mm_madd_epi16(src_3, coeff);
        __m128i dst_4 = _mm_madd_epi16(src_4, coeff);

        result_1 = _mm_add_epi32(result_1, dst_1);
        result_2 = _mm_add_epi32(result_2, dst_2);
        result_3 = _mm_add_epi32(result_3, dst_3);
        result_4 = _mm_add_epi32(result_4, dst_4);
      }
      
      if (notMod2) { // do last odd row
        __m128i src_p = load(reinterpret_cast<const __m128i*>(src_ptr+pitch_table[sizeMod2]+x));

        __m128i src_l = _mm_unpacklo_epi8(src_p, zero);
        __m128i src_h = _mm_unpackhi_epi8(src_p, zero);

        __m128i coeff = _mm_set1_epi16(current_coeff[sizeMod2]);

        __m128i dst_ll = _mm_mullo_epi16(src_l, coeff);   // Multiply by coefficient
        __m128i dst_lh = _mm_mulhi_epi16(src_l, coeff);
        __m128i dst_hl = _mm_mullo_epi16(src_h, coeff);
        __m128i dst_hh = _mm_mulhi_epi16(src_h, coeff);

        __m128i dst_1 = _mm_unpacklo_epi16(dst_ll, dst_lh); // Unpack to 32-bit integer
        __m128i dst_2 = _mm_unpackhi_epi16(dst_ll, dst_lh);
        __m128i dst_3 = _mm_unpacklo_epi16(dst_hl, dst_hh);
        __m128i dst_4 = _mm_unpackhi_epi16(dst_hl, dst_hh);

        result_1 = _mm_add_epi32(result_1, dst_1);
        result_2 = _mm_add_epi32(result_2, dst_2);
        result_3 = _mm_add_epi32(result_3, dst_3);
        result_4 = _mm_add_epi32(result_4, dst_4);
      }
      
      // Divide by 16348 (FPRound)
      result_1  = _mm_srai_epi32(result_1, 14);
      result_2  = _mm_srai_epi32(result_2, 14);
      result_3  = _mm_srai_epi32(result_3, 14);
      result_4  = _mm_srai_epi32(result_4, 14);

      // Pack and store
      __m128i result_l = _mm_packs_epi32(result_1, result_2);
      __m128i result_h = _mm_packs_epi32(result_3, result_4);
      __m128i result   = _mm_packus_epi16(result_l, result_h);

      _mm_store_si128(reinterpret_cast<__m128i*>(dst+x), result);
    }

    // Leftover
    for (int x = wMod16; x < width; x++)
	{
      int result = 0;

      for (int i = 0; i < filter_size; i++)
        result += (src_ptr+pitch_table[i])[x] * current_coeff[i];
	  result = (result+8192) >> 14;
      result = result > 255 ? 255 : result < 0 ? 0 : result;
      dst[x] = (BYTE) result;
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

template<SSELoader load>
static void resize_v_ssse3_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width,  int MinY, int MaxY, const int* pitch_table, const void* storage)
{
  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient + filter_size*MinY;
  
  int wMod16 = (width >> 4) << 4;

  __m128i zero = _mm_setzero_si128();
  __m128i coeff_unpacker = _mm_set_epi8(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);

  for (int y = MinY; y < MaxY; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + pitch_table[offset];

    for (int x = 0; x < wMod16; x+=16) {
      __m128i result_l = _mm_set1_epi16(32); // Init. with rounder ((1 << 6)/2 = 32)
      __m128i result_h = result_l;

      const BYTE* src2_ptr = src_ptr+x;
      
      for (int i = 0; i < filter_size; i++) {
        __m128i src_p = load(reinterpret_cast<const __m128i*>(src2_ptr));

        __m128i src_l = _mm_unpacklo_epi8(src_p, zero);
        __m128i src_h = _mm_unpackhi_epi8(src_p, zero);

        src_l = _mm_slli_epi16(src_l, 7);
        src_h = _mm_slli_epi16(src_h, 7);

        __m128i coeff = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(current_coeff+i));
                coeff = _mm_shuffle_epi8(coeff, coeff_unpacker);

        __m128i dst_l = _mm_mulhrs_epi16(src_l, coeff);   // Multiply by coefficient (SSSE3)
        __m128i dst_h = _mm_mulhrs_epi16(src_h, coeff);

        result_l = _mm_add_epi16(result_l, dst_l);
        result_h = _mm_add_epi16(result_h, dst_h);

        src2_ptr += src_pitch;
      }

      // Divide by 64
      result_l  = _mm_srai_epi16(result_l, 6);
      result_h  = _mm_srai_epi16(result_h, 6);

      // Pack and store
      __m128i result   = _mm_packus_epi16(result_l, result_h);

      _mm_store_si128(reinterpret_cast<__m128i*>(dst+x), result);
    }

    // Leftover
    for (int x = wMod16; x < width; x++)
	{
      int result = 0;

      for (int i = 0; i < filter_size; i++)
        result += (src_ptr+pitch_table[i])[x] * current_coeff[i];
	  result = (result+8192) >> 14;
      result = result > 255 ? 255 : result < 0 ? 0 : result;
      dst[x] = (BYTE) result;
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

__forceinline static void resize_v_create_pitch_table(int* table, int pitch, int height, uint8_t pixel_size)
{
  switch(pixel_size)
  {
	case 2 : pitch>>=1; break;
	case 4 : pitch>>=2; break;
	default : ;
  }
  table[0] = 0;
  for (int i = 1; i < height; i++)
    table[i] = table[i-1]+pitch;
}


/***************************************
 ********* Horizontal Resizer** ********
 ***************************************/

static void resize_h_pointresize(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height) {
  int wMod4 = (width >> 2) << 2;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod4; x+=4) {
#define pixel(a) src[program->pixel_offset[x+a]]
      unsigned int data = (pixel(3) << 24) + (pixel(2) << 16) + (pixel(1) << 8) + pixel(0);
#undef pixel
      *((unsigned int *)(dst+x)) = data;
    }

    for (int x = wMod4; x < width; x++) {
      dst[x] = src[program->pixel_offset[x]];
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

static void resize_h_prepare_coeff_8(ResamplingProgram* p,IScriptEnvironment* env) {
  int filter_size = AlignNumber(p->filter_size, 8);
  short* new_coeff = (short*) _aligned_malloc(sizeof(short) * p->target_size * filter_size, 64);
  float* new_coeff_float = (float*) _aligned_malloc(sizeof(float) * p->target_size * filter_size, 64);
  if ((new_coeff==NULL) || (new_coeff_float==NULL)) {
	myalignedfree(new_coeff_float);
    myalignedfree(new_coeff);
    env->ThrowError("ResizeMT: Could not reserve memory in a resampler.");
  }

  memset(new_coeff, 0, sizeof(short) * p->target_size * filter_size);
  memset(new_coeff_float, 0, sizeof(float) * p->target_size * filter_size);
  
  // Copy coeff
  short *dst = new_coeff, *src = p->pixel_coefficient;
  float *dst_f = new_coeff_float, *src_f = p->pixel_coefficient_float;
  for (int i = 0; i < p->target_size; i++) {
    for (int j = 0; j < p->filter_size; j++) {
      dst[j] = src[j];
      dst_f[j] = src_f[j];
    }

    dst += filter_size;
    src += p->filter_size;
  }

  myalignedfree(p->pixel_coefficient_float);
  myalignedfree(p->pixel_coefficient);
  p->pixel_coefficient = new_coeff;
  p->pixel_coefficient_float = new_coeff_float;
}


static void resize_h_c_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height)
{
  int filter_size = program->filter_size;

  short *current_coeff=program->pixel_coefficient;

  for (int x = 0; x < width; x++)
  {
    int begin = program->pixel_offset[x];
	int y_src_pitch=0,y_dst_pitch=0;

    for (int y = 0; y < height; y++)
	{
      // todo: check whether int result is enough for 16 bit samples (can an int overflow because of 16384 scale or really need __int64?)
      int result = 0;

      for (int i = 0; i < filter_size; i++)
		result += (src+y_src_pitch)[(begin+i)] * current_coeff[i];
	  result = (result + 8192) >> 14;
	  result = result > 255 ? 255 : result < 0 ? 0 : result;
	  (dst + y_dst_pitch)[x] = (BYTE)result;

	  y_dst_pitch+=dst_pitch;
	  y_src_pitch+=src_pitch;
    }
    current_coeff += filter_size;
  }
}


static void resize_h_c_planar_s(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height)
{
  int filter_size = program->filter_size;

  short *current_coeff=program->pixel_coefficient;

  src_pitch>>=1;
  dst_pitch>>=1;

  uint16_t* src0 = (uint16_t*)src;
  uint16_t* dst0 = (uint16_t*)dst;

  for (int x = 0; x < width; x++)
  {
    int begin = program->pixel_offset[x];
	int y_src_pitch=0,y_dst_pitch=0;

    for (int y = 0; y < height; y++)
	{
      __int64 result = 0;

      for (int i = 0; i < filter_size; i++)
        result += (src0+y_src_pitch)[(begin+i)] * current_coeff[i];
	  result = (result + 8192) >> 14;
	  result = result > 65535 ? 65535 : result < 0 ? 0 : result;
      (dst0 + y_dst_pitch)[x] = (uint16_t)result;

	  y_dst_pitch+=dst_pitch;
	  y_src_pitch+=src_pitch;
    }
    current_coeff += filter_size;
  }
}


static void resize_h_c_planar_f(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height)
{
  int filter_size = program->filter_size;

  float *current_coeff=program->pixel_coefficient_float;

  src_pitch = src_pitch >> 2;
  dst_pitch = dst_pitch >> 2;

  float* src0 = (float*)src;
  float* dst0 = (float*)dst;

  for (int x = 0; x < width; x++)
  {
    int begin = program->pixel_offset[x];
	int y_src_pitch=0,y_dst_pitch=0;

    for (int y = 0; y < height; y++)
	{
      float result = 0;

      for (int i = 0; i < filter_size; i++)
        result += (src0+y_src_pitch)[(begin+i)] * current_coeff[i];
      (dst0 + y_dst_pitch)[x] = result;

	  y_dst_pitch+=dst_pitch;
	  y_src_pitch+=src_pitch;
    }
    current_coeff += filter_size;
  }
}




static void resizer_h_ssse3_generic(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height) {
  int filter_size = AlignNumber(program->filter_size, 8) >> 3;
  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height; y++)
  {
    short* current_coeff = program->pixel_coefficient;
    for (int x = 0; x < width; x+=4)
	{
      __m128i result1 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result2 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result3 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result4 = _mm_setr_epi32(8192, 0, 0, 0);

      int begin1 = program->pixel_offset[x+0];
      int begin2 = program->pixel_offset[x+1];
      int begin3 = program->pixel_offset[x+2];
      int begin4 = program->pixel_offset[x+3];

	  for (int i = 0; i < filter_size; i++)
	  {
	    __m128i data, coeff, current_result;
		data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+begin1+i*8));
        data = _mm_unpacklo_epi8(data, zero);
	    coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
		current_result = _mm_madd_epi16(data, coeff);
        result1 = _mm_add_epi32(result1, current_result);
			
	    current_coeff += 8;		
	  }

      for (int i = 0; i < filter_size; i++)
	  {
		__m128i data, coeff, current_result;
        data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+begin2+i*8));
	    data = _mm_unpacklo_epi8(data, zero);
		coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
        current_result = _mm_madd_epi16(data, coeff);
	    result2 = _mm_add_epi32(result2, current_result);

		current_coeff += 8;
      }

      for (int i = 0; i < filter_size; i++)
	  {
		__m128i data, coeff, current_result;
        data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+begin3+i*8));
	    data = _mm_unpacklo_epi8(data, zero);
		coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
        current_result = _mm_madd_epi16(data, coeff);
	    result3 = _mm_add_epi32(result3, current_result);

		current_coeff += 8;
	  }

      for (int i = 0; i < filter_size; i++)
	  {
		__m128i data, coeff, current_result;
        data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+begin4+i*8));
		data = _mm_unpacklo_epi8(data, zero);
	    coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
        current_result = _mm_madd_epi16(data, coeff);
	    result4 = _mm_add_epi32(result4, current_result);

        current_coeff += 8;
	  }

      __m128i result12 = _mm_hadd_epi32(result1, result2);
      __m128i result34 = _mm_hadd_epi32(result3, result4);
      __m128i result = _mm_hadd_epi32(result12, result34);

      result = _mm_srai_epi32(result, 14);

      result = _mm_packs_epi32(result, zero);
      result = _mm_packus_epi16(result, zero);

      *((int*)(dst+x)) = _mm_cvtsi128_si32(result);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

static void resizer_h_ssse3_8(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height) {
  int filter_size = AlignNumber(program->filter_size, 8) / 8;

  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height; y++) {
    short* current_coeff = program->pixel_coefficient;
    for (int x = 0; x < width; x+=4) {
      __m128i result1 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result2 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result3 = _mm_setr_epi32(8192, 0, 0, 0);
      __m128i result4 = _mm_setr_epi32(8192, 0, 0, 0);

      int begin1 = program->pixel_offset[x+0];
      int begin2 = program->pixel_offset[x+1];
      int begin3 = program->pixel_offset[x+2];
      int begin4 = program->pixel_offset[x+3];

      __m128i data, coeff, current_result;

      // Unroll 1
      data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+begin1));
      data = _mm_unpacklo_epi8(data, zero);
      coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
      current_result = _mm_madd_epi16(data, coeff);
      result1 = _mm_add_epi32(result1, current_result);

      current_coeff += 8;

      // Unroll 2
      data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+begin2));
      data = _mm_unpacklo_epi8(data, zero);
      coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
      current_result = _mm_madd_epi16(data, coeff);
      result2 = _mm_add_epi32(result2, current_result);

      current_coeff += 8;

      // Unroll 3
      data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+begin3));
      data = _mm_unpacklo_epi8(data, zero);
      coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
      current_result = _mm_madd_epi16(data, coeff);
      result3 = _mm_add_epi32(result3, current_result);

      current_coeff += 8;

      // Unroll 4
      data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+begin4));
      data = _mm_unpacklo_epi8(data, zero);
      coeff = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff));
      current_result = _mm_madd_epi16(data, coeff);
      result4 = _mm_add_epi32(result4, current_result);

      current_coeff += 8;

      // Combine
      __m128i result12 = _mm_hadd_epi32(result1, result2);
      __m128i result34 = _mm_hadd_epi32(result3, result4);
      __m128i result = _mm_hadd_epi32(result12, result34);

      result = _mm_srai_epi32(result, 14);

      result = _mm_packs_epi32(result, zero);
      result = _mm_packus_epi16(result, zero);

      *((int*)(dst+x)) = _mm_cvtsi128_si32(result);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}



static int num_processors()
{
#ifdef _DEBUG
	return 1;
#else
	int pcount = 0;
	ULONG_PTR p_aff=0, s_aff=0;
	GetProcessAffinityMask(GetCurrentProcess(), &p_aff, &s_aff);
	for(; p_aff != 0; p_aff>>=1) 
		pcount += (p_aff&1);
	return pcount;
#endif
}



FilteredResizeH::FilteredResizeH( PClip _child, double subrange_left, double subrange_width,
                                  int target_width, int _threads, bool _avsp, ResamplingFunction* func, IScriptEnvironment* env )
  : GenericVideoFilter(_child),
  resampling_program_luma(NULL), resampling_program_chroma(NULL),
  filter_storage_luma(NULL), filter_storage_chroma(NULL),
  threads(_threads),avsp(_avsp)
{
  src_width  = vi.width;
  src_height = vi.height;
  dst_width  = target_width;
  dst_height = vi.height;

  if (avsp)
  {
	pixelsize = (uint8_t)vi.ComponentSize(); // AVS16
	grey = vi.IsY8() || vi.IsColorSpace(VideoInfo::CS_Y16) || vi.IsColorSpace(VideoInfo::CS_Y32);	  
  }
  else
  {
	pixelsize = 1;
	grey = vi.IsY8();
  }  

	bool ok,def_affinity;
	int16_t i;

	for (i=0; i<MAX_MT_THREADS; i++)
	{
		MT_Thread[i].pClass=NULL;
		MT_Thread[i].f_process=0;
		MT_Thread[i].thread_Id=(uint8_t)i;
		MT_Thread[i].jobFinished=NULL;
		MT_Thread[i].nextJob=NULL;
		thds[i]=NULL;
	}
	ghMutex=NULL;

	CPUs_number=(uint8_t)num_processors();
	if (CPUs_number>MAX_MT_THREADS) CPUs_number=MAX_MT_THREADS;

	if (vi.height>=32)
	{
		if (threads==0) threads_number=CPUs_number;
		else threads_number=(uint8_t)threads;
	}
	else threads_number=1;

	const int shift_w = (!grey && vi.IsPlanar()) ? vi.GetPlaneWidthSubsampling(PLANAR_U) : 0;
	const int shift_h = (!grey && vi.IsPlanar()) ? vi.GetPlaneHeightSubsampling(PLANAR_U) : 0;

	const int src_width = vi.IsPlanar() ? vi.width : vi.BytesFromPixels(vi.width);
	const int dst_width = vi.IsPlanar() ? target_width : vi.BytesFromPixels(target_width);
	
	threads_number=CreateMTData(threads_number,src_width,vi.height,dst_width,vi.height,shift_w,shift_h);
	if (threads_number<=CPUs_number) def_affinity=true;
	else def_affinity=false;

	ghMutex=CreateMutex(NULL,FALSE,NULL);
	if (ghMutex==NULL) env->ThrowError("ResizeMT: Unable to create Mutex !");

	if (threads_number>1)
	{
		ok=true;
		i=0;
		while ((i<threads_number) && ok)
		{
			MT_Thread[i].pClass=this;
			MT_Thread[i].f_process=0;
			MT_Thread[i].jobFinished=CreateEvent(NULL,TRUE,TRUE,NULL);
			MT_Thread[i].nextJob=CreateEvent(NULL,TRUE,FALSE,NULL);
			ok=ok && ((MT_Thread[i].jobFinished!=NULL) && (MT_Thread[i].nextJob!=NULL));
			i++;
		}
		if (!ok)
		{
			FreeData();
			env->ThrowError("ResizeMT: Unable to create events !");
		}

		DWORD_PTR dwpProcessAffinityMask;
		DWORD_PTR dwpSystemAffinityMask;
		DWORD_PTR dwpThreadAffinityMask=1;

		GetProcessAffinityMask(GetCurrentProcess(), &dwpProcessAffinityMask, &dwpSystemAffinityMask);

		ok=true;
		i=0;
		while ((i<threads_number) && ok)
		{
			if (def_affinity)
			{
				if ((dwpProcessAffinityMask & dwpThreadAffinityMask)!=0)
				{
					thds[i]=CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)StaticThreadpoolH,&MT_Thread[i],CREATE_SUSPENDED,&tids[i]);
					ok=ok && (thds[i]!=NULL);
					if (ok)
					{
						SetThreadAffinityMask(thds[i],dwpThreadAffinityMask);
						ResumeThread(thds[i]);
					}
					i++;
				}
				dwpThreadAffinityMask<<=1;
			}
			else
			{
				thds[i]=CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)StaticThreadpoolH,&MT_Thread[i],0,&tids[i]);
				ok=ok && (thds[i]!=NULL);
				i++;
			}
		}
		if (!ok)
		{
			FreeData();
			env->ThrowError("ResizeMT: Unable to create threads pool !");
		}
	}  
  
  // Main resampling program
  resampling_program_luma = func->GetResamplingProgram(vi.width, subrange_left, subrange_width, target_width, env);
  if (vi.IsPlanar() && !grey) {
    const int div   = 1 << shift_w;


    resampling_program_chroma = func->GetResamplingProgram(
      vi.width       >> shift_w,
      subrange_left   / div,
      subrange_width  / div,
      target_width   >> shift_w,
	  env);
  }
  
  
  // Plannar + SSSE3 = use new horizontal resizer routines
  resampler_h_luma = GetResampler(env->GetCPUFlags(), true, pixelsize, resampling_program_luma,env);

  if (!grey) resampler_h_chroma = GetResampler(env->GetCPUFlags(), true, pixelsize, resampling_program_chroma,env);
  
  // Change target video info size
  vi.width = target_width;
}



int __stdcall FilteredResizeH::SetCacheHints(int cachehints,int frame_range)
{
  switch (cachehints)
  {
  case CACHE_DONT_CACHE_ME:
    return 1;
  case CACHE_GET_MTMODE:
    return MT_MULTI_INSTANCE;
  default:
    return 0;
  }
}


uint8_t FilteredResizeH::CreateMTData(uint8_t max_threads,int32_t src_size_x,int32_t src_size_y,int32_t dst_size_x,int32_t dst_size_y,int UV_w,int UV_h)
{
	int32_t _y_min,_dh;

	if ((max_threads<=1) || (max_threads>threads_number))
	{
		MT_Data[0].top=true;
		MT_Data[0].bottom=true;
		MT_Data[0].src_Y_h_min=0;
		MT_Data[0].dst_Y_h_min=0;
		MT_Data[0].src_Y_h_max=src_size_y;
		MT_Data[0].dst_Y_h_max=dst_size_y;
		MT_Data[0].src_UV_h_min=0;
		MT_Data[0].dst_UV_h_min=0;
		if (UV_h>0)
		{
			MT_Data[0].src_UV_h_max=src_size_y >> UV_h;
			MT_Data[0].dst_UV_h_max=dst_size_y >> UV_h;
		}
		else
		{
			MT_Data[0].src_UV_h_max=src_size_y;
			MT_Data[0].dst_UV_h_max=dst_size_y;
		}
		MT_Data[0].src_Y_w=src_size_x;
		MT_Data[0].dst_Y_w=dst_size_x;
		if (UV_w>0)
		{
			MT_Data[0].src_UV_w=src_size_x >> UV_w;
			MT_Data[0].dst_UV_w=dst_size_x >> UV_w;
		}
		else
		{
			MT_Data[0].src_UV_w=src_size_x;
			MT_Data[0].dst_UV_w=dst_size_x;
		}
		return(1);
	}

	int32_t src_dh_Y,src_dh_UV,dst_dh_Y,dst_dh_UV;
	int32_t h_y;
	uint8_t i,max=0;

	dst_dh_Y=(dst_size_y+(uint32_t)max_threads-1)/(uint32_t)max_threads;
	if (dst_dh_Y<16) dst_dh_Y=16;
	if ((dst_dh_Y & 3)!=0) dst_dh_Y=((dst_dh_Y+3) >> 2) << 2;

	if (src_size_y==dst_size_y) src_dh_Y=dst_dh_Y;
	else
	{
		src_dh_Y=(src_size_y+(uint32_t)max_threads-1)/(uint32_t)max_threads;
		if (src_dh_Y<16) src_dh_Y=16;
		if ((src_dh_Y & 3)!=0) src_dh_Y=((src_dh_Y+3) >> 2) << 2;
	}

	if (src_size_y<dst_size_y)
	{
		_y_min=src_size_y;
		_dh=src_dh_Y;
	}
	else
	{
		_y_min=dst_size_y;
		_dh=dst_dh_Y;
	}
	h_y=0;
	while (h_y<(_y_min-16))
	{
		max++;
		h_y+=_dh;
	}

	if (max==1)
	{
		MT_Data[0].top=true;
		MT_Data[0].bottom=true;
		MT_Data[0].src_Y_h_min=0;
		MT_Data[0].dst_Y_h_min=0;
		MT_Data[0].src_Y_h_max=src_size_y;
		MT_Data[0].dst_Y_h_max=dst_size_y;
		MT_Data[0].src_UV_h_min=0;
		MT_Data[0].dst_UV_h_min=0;
		if (UV_h>0)
		{
			MT_Data[0].src_UV_h_max=src_size_y >> UV_h;
			MT_Data[0].dst_UV_h_max=dst_size_y >> UV_h;
		}
		else
		{
			MT_Data[0].src_UV_h_max=src_size_y;
			MT_Data[0].dst_UV_h_max=dst_size_y;
		}
		MT_Data[0].src_Y_w=src_size_x;
		MT_Data[0].dst_Y_w=dst_size_x;
		if (UV_w>0)
		{
			MT_Data[0].src_UV_w=src_size_x >> UV_w;
			MT_Data[0].dst_UV_w=dst_size_x >> UV_w;
		}
		else
		{
			MT_Data[0].src_UV_w=src_size_x;
			MT_Data[0].dst_UV_w=dst_size_x;
		}
		return(1);
	}

	src_dh_UV= (UV_h>0) ? src_dh_Y>>UV_h : src_dh_Y;
	dst_dh_UV= (UV_h>0) ? dst_dh_Y>>UV_h : dst_dh_Y;

	MT_Data[0].top=true;
	MT_Data[0].bottom=false;
	MT_Data[0].src_Y_h_min=0;
	MT_Data[0].src_Y_h_max=src_dh_Y;
	MT_Data[0].dst_Y_h_min=0;
	MT_Data[0].dst_Y_h_max=dst_dh_Y;
	MT_Data[0].src_UV_h_min=0;
	MT_Data[0].src_UV_h_max=src_dh_UV;
	MT_Data[0].dst_UV_h_min=0;
	MT_Data[0].dst_UV_h_max=dst_dh_UV;

	i=1;
	while (i<max)
	{
		MT_Data[i].top=false;
		MT_Data[i].bottom=false;
		MT_Data[i].src_Y_h_min=MT_Data[i-1].src_Y_h_max;
		MT_Data[i].src_Y_h_max=MT_Data[i].src_Y_h_min+src_dh_Y;
		MT_Data[i].dst_Y_h_min=MT_Data[i-1].dst_Y_h_max;
		MT_Data[i].dst_Y_h_max=MT_Data[i].dst_Y_h_min+dst_dh_Y;
		MT_Data[i].src_UV_h_min=MT_Data[i-1].src_UV_h_max;
		MT_Data[i].src_UV_h_max=MT_Data[i].src_UV_h_min+src_dh_UV;
		MT_Data[i].dst_UV_h_min=MT_Data[i-1].dst_UV_h_max;
		MT_Data[i].dst_UV_h_max=MT_Data[i].dst_UV_h_min+dst_dh_UV;
		i++;
	}

	MT_Data[max-1].bottom=true;
	MT_Data[max-1].src_Y_h_max=src_size_y;
	MT_Data[max-1].dst_Y_h_max=dst_size_y;
	if (UV_h>0)
	{
		MT_Data[max-1].src_UV_h_max=src_size_y >> UV_h;
		MT_Data[max-1].dst_UV_h_max=dst_size_y >> UV_h;
	}
	else
	{
		MT_Data[max-1].src_UV_h_max=src_size_y;
		MT_Data[max-1].dst_UV_h_max=dst_size_y;
	}

	for (i=0; i<max; i++)
	{
		MT_Data[i].src_Y_w=src_size_x;
		MT_Data[i].dst_Y_w=dst_size_x;
		if (UV_w>0)
		{
			MT_Data[i].src_UV_w=src_size_x >> UV_w;
			MT_Data[i].dst_UV_w=dst_size_x >> UV_w;
		}
		else
		{
			MT_Data[i].src_UV_w=src_size_x;
			MT_Data[i].dst_UV_w=dst_size_x;
		}
	}

	return(max);
}


void FilteredResizeH::ResamplerLumaMT(uint8_t thread_num)
{
	const MT_Data_Info mt_data_inf=MT_Data[thread_num];

	resampler_h_luma(mt_data_inf.dst1,mt_data_inf.src1,mt_data_inf.dst_pitch1,mt_data_inf.src_pitch1,
		mt_data_inf.resampling_program_luma,mt_data_inf.dst_Y_w,mt_data_inf.dst_Y_h_max-mt_data_inf.dst_Y_h_min);
}


void FilteredResizeH::ResamplerUChromaMT(uint8_t thread_num)
{
	const MT_Data_Info mt_data_inf=MT_Data[thread_num];

	resampler_h_luma(mt_data_inf.dst2,mt_data_inf.src2,mt_data_inf.dst_pitch2,mt_data_inf.src_pitch2,
		mt_data_inf.resampling_program_chroma,mt_data_inf.dst_UV_w,mt_data_inf.dst_UV_h_max-mt_data_inf.dst_UV_h_min);
}


void FilteredResizeH::ResamplerVChromaMT(uint8_t thread_num)
{
	const MT_Data_Info mt_data_inf=MT_Data[thread_num];

	resampler_h_luma(mt_data_inf.dst3,mt_data_inf.src3,mt_data_inf.dst_pitch3,mt_data_inf.src_pitch3,
		mt_data_inf.resampling_program_chroma,mt_data_inf.dst_UV_w,mt_data_inf.dst_UV_h_max-mt_data_inf.dst_UV_h_min);
}


DWORD WINAPI FilteredResizeH::StaticThreadpoolH( LPVOID lpParam )
{
	const MT_Data_Thread *data=(const MT_Data_Thread *)lpParam;
	FilteredResizeH *ptrClass=(FilteredResizeH *)data->pClass;
	
	while (true)
	{
		WaitForSingleObject(data->nextJob,INFINITE);
		switch(data->f_process)
		{
			case 1 : ptrClass->ResamplerLumaMT(data->thread_Id);
				break;
			case 2 : ptrClass->ResamplerUChromaMT(data->thread_Id);
				break;
			case 3 : ptrClass->ResamplerVChromaMT(data->thread_Id);
				break;
			case 255 : return(0); break;
			default : ;
		}
		ResetEvent(data->nextJob);
		SetEvent(data->jobFinished);
	}
}


PVideoFrame __stdcall FilteredResizeH::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrame(vi);
  
  const int src_pitch_Y = src->GetPitch();
  const int dst_pitch_Y = dst->GetPitch();
  const BYTE* srcp_Y = src->GetReadPtr();
        BYTE* dstp_Y = dst->GetWritePtr();

	const int src_pitch_U = (!grey && vi.IsPlanar()) ? src->GetPitch(PLANAR_U) : 0;
	const int dst_pitch_U = (!grey && vi.IsPlanar()) ? dst->GetPitch(PLANAR_U) : 0;
	const BYTE* srcp_U = (!grey && vi.IsPlanar()) ? src->GetReadPtr(PLANAR_U) : NULL;
	BYTE* dstp_U = (!grey && vi.IsPlanar()) ? dst->GetWritePtr(PLANAR_U) : NULL;

	const int src_pitch_V = (!grey && vi.IsPlanar()) ? src->GetPitch(PLANAR_V) : 0;
	const int dst_pitch_V = (!grey && vi.IsPlanar()) ? dst->GetPitch(PLANAR_V) : 0;
	const BYTE* srcp_V = (!grey && vi.IsPlanar()) ? src->GetReadPtr(PLANAR_V) : NULL;
	BYTE* dstp_V = (!grey && vi.IsPlanar()) ? dst->GetWritePtr(PLANAR_V) : NULL;


  WaitForSingleObject(ghMutex,INFINITE);
  
	for(uint8_t i=0; i<threads_number; i++)
	{
		MT_Data[i].src1=srcp_Y+(MT_Data[i].src_Y_h_min*src_pitch_Y);
		MT_Data[i].src2=srcp_U+(MT_Data[i].src_UV_h_min*src_pitch_U);
		MT_Data[i].src3=srcp_V+(MT_Data[i].src_UV_h_min*src_pitch_V);
		MT_Data[i].src_pitch1=src_pitch_Y;
		MT_Data[i].src_pitch2=src_pitch_U;
		MT_Data[i].src_pitch3=src_pitch_V;
		MT_Data[i].dst1=dstp_Y+(MT_Data[i].dst_Y_h_min*dst_pitch_Y);
		MT_Data[i].dst2=dstp_U+(MT_Data[i].dst_UV_h_min*dst_pitch_U);
		MT_Data[i].dst3=dstp_V+(MT_Data[i].dst_UV_h_min*dst_pitch_V);
		MT_Data[i].dst_pitch1=dst_pitch_Y;
		MT_Data[i].dst_pitch2=dst_pitch_U;
		MT_Data[i].dst_pitch3=dst_pitch_V;
		MT_Data[i].filter_storage_luma=filter_storage_luma;
		MT_Data[i].resampling_program_luma=resampling_program_luma;
		MT_Data[i].resampling_program_chroma=resampling_program_chroma;
		MT_Data[i].filter_storage_chromaU=filter_storage_chroma;
		MT_Data[i].filter_storage_chromaV=filter_storage_chroma;
	}


	if (threads_number>1)
	{
		uint8_t f_proc;

		f_proc=1;

		for(uint8_t i=0; i<threads_number; i++)
		{
			MT_Thread[i].f_process=f_proc;
			ResetEvent(MT_Thread[i].jobFinished);
			SetEvent(MT_Thread[i].nextJob);
		}
		for(uint8_t i=0; i<threads_number; i++)
			WaitForSingleObject(MT_Thread[i].jobFinished,INFINITE);

		if (!grey && vi.IsPlanar())
		{
			f_proc=2;

			for(uint8_t i=0; i<threads_number; i++)
			{
				MT_Thread[i].f_process=f_proc;
				ResetEvent(MT_Thread[i].jobFinished);
				SetEvent(MT_Thread[i].nextJob);
			}
			for(uint8_t i=0; i<threads_number; i++)
				WaitForSingleObject(MT_Thread[i].jobFinished,INFINITE);

			f_proc=3;

			for(uint8_t i=0; i<threads_number; i++)
			{
				MT_Thread[i].f_process=f_proc;
				ResetEvent(MT_Thread[i].jobFinished);
				SetEvent(MT_Thread[i].nextJob);
			}
			for(uint8_t i=0; i<threads_number; i++)
				WaitForSingleObject(MT_Thread[i].jobFinished,INFINITE);
		}

		for(uint8_t i=0; i<threads_number; i++)
			MT_Thread[i].f_process=0;
	}
	else
	{
		// Do resizing
		ResamplerLumaMT(0);
    
		if (!grey && vi.IsPlanar())
		{
			// Plane U resizing   
			ResamplerUChromaMT(0);

			// Plane V resizing
			ResamplerVChromaMT(0);
		}
	}

	ReleaseMutex(ghMutex);  
	
  return dst;
}


ResamplerH FilteredResizeH::GetResampler(int CPU, bool aligned, int pixelsize, ResamplingProgram* program, IScriptEnvironment* env)
{
  if (pixelsize == 1)
  {
  if (CPU & CPUF_SSSE3) {
    resize_h_prepare_coeff_8(program,env);
    if (program->filter_size > 8)
      return resizer_h_ssse3_generic;
    else
      return resizer_h_ssse3_8;
  }
    else { // C version
      return resize_h_c_planar;
}
  }
  else if (pixelsize == 2) { // todo: non_c
    return resize_h_c_planar_s;
  } else { //if (pixelsize == 4)
    return resize_h_c_planar_f;
  }
}

void FilteredResizeH::FreeData(void) 
{
	int16_t i;

  if (resampling_program_luma!=NULL)   { delete resampling_program_luma; }
  if (resampling_program_chroma!=NULL) { delete resampling_program_chroma; }
	
  myalignedfree(filter_storage_luma);
  myalignedfree(filter_storage_chroma);

	if (threads_number>1)
	{
		for (i=threads_number-1; i>=0; i--)
		{
			if (thds[i]!=NULL)
			{
				MT_Thread[i].f_process=255;
				SetEvent(MT_Thread[i].nextJob);
				WaitForSingleObject(thds[i],INFINITE);
				myCloseHandle(thds[i]);
			}
		}
		for (i=threads_number-1; i>=0; i--)
		{
			myCloseHandle(MT_Thread[i].nextJob);
			myCloseHandle(MT_Thread[i].jobFinished);
		}
	}
	myCloseHandle(ghMutex);
}

FilteredResizeH::~FilteredResizeH(void)
{
	FreeData();
}


/***************************************
 ***** Filtered Resize - Vertical ******
 ***************************************/

FilteredResizeV::FilteredResizeV( PClip _child, double subrange_top, double subrange_height,
                                  int target_height, int _threads, bool _avsp, ResamplingFunction* func, IScriptEnvironment* env )
  : GenericVideoFilter(_child),
    resampling_program_luma(NULL), resampling_program_chroma(NULL),
    src_pitch_table_luma(NULL), src_pitch_table_chromaU(NULL), src_pitch_table_chromaV(NULL),
    src_pitch_luma(-1), src_pitch_chromaU(-1), src_pitch_chromaV(-1),
    filter_storage_luma_aligned(NULL), filter_storage_luma_unaligned(NULL),
    filter_storage_chroma_aligned(NULL), filter_storage_chroma_unaligned(NULL), threads(_threads),avsp(_avsp)
{
	bool ok,def_affinity;
	int16_t i;

  if (avsp)
  {
	pixelsize = (uint8_t)vi.ComponentSize(); // AVS16
	grey = vi.IsY8() || vi.IsColorSpace(VideoInfo::CS_Y16) || vi.IsColorSpace(VideoInfo::CS_Y32);	  
  }
  else
  {
	pixelsize = 1;
	grey = vi.IsY8();
  }  
	
	
	for (i=0; i<MAX_MT_THREADS; i++)
	{
		MT_Thread[i].pClass=NULL;
		MT_Thread[i].f_process=0;
		MT_Thread[i].thread_Id=(uint8_t)i;
		MT_Thread[i].jobFinished=NULL;
		MT_Thread[i].nextJob=NULL;
		thds[i]=NULL;
	}
	ghMutex=NULL;

	CPUs_number=(uint8_t)num_processors();
	if (CPUs_number>MAX_MT_THREADS) CPUs_number=MAX_MT_THREADS;

	if (vi.height>=32)
	{
		if (threads==0) threads_number=CPUs_number;
		else threads_number=(uint8_t)threads;
	}
	else threads_number=1;

	const int shift_w = (!grey && vi.IsPlanar()) ? vi.GetPlaneWidthSubsampling(PLANAR_U) : 0;
	const int shift_h = (!grey && vi.IsPlanar()) ? vi.GetPlaneHeightSubsampling(PLANAR_U) : 0;

	const int work_width = vi.IsPlanar() ? vi.width : vi.BytesFromPixels(vi.width);
	
	threads_number=CreateMTData(threads_number,work_width,vi.height,work_width,target_height,shift_w,shift_h);
	if (threads_number<=CPUs_number) def_affinity=true;
	else def_affinity=false;

	ghMutex=CreateMutex(NULL,FALSE,NULL);
	if (ghMutex==NULL) env->ThrowError("ResizeMT: Unable to create Mutex !");

	if (threads_number>1)
	{
		ok=true;
		i=0;
		while ((i<threads_number) && ok)
		{
			MT_Thread[i].pClass=this;
			MT_Thread[i].f_process=0;
			MT_Thread[i].jobFinished=CreateEvent(NULL,TRUE,TRUE,NULL);
			MT_Thread[i].nextJob=CreateEvent(NULL,TRUE,FALSE,NULL);
			ok=ok && ((MT_Thread[i].jobFinished!=NULL) && (MT_Thread[i].nextJob!=NULL));
			i++;
		}
		if (!ok)
		{
			FreeData();
			env->ThrowError("ResizeMT: Unable to create events !");
		}

		DWORD_PTR dwpProcessAffinityMask;
		DWORD_PTR dwpSystemAffinityMask;
		DWORD_PTR dwpThreadAffinityMask=1;

		GetProcessAffinityMask(GetCurrentProcess(), &dwpProcessAffinityMask, &dwpSystemAffinityMask);

		ok=true;
		i=0;
		while ((i<threads_number) && ok)
		{
			if (def_affinity)
			{
				if ((dwpProcessAffinityMask & dwpThreadAffinityMask)!=0)
				{
					thds[i]=CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)StaticThreadpoolV,&MT_Thread[i],CREATE_SUSPENDED,&tids[i]);
					ok=ok && (thds[i]!=NULL);
					if (ok)
					{
						SetThreadAffinityMask(thds[i],dwpThreadAffinityMask);
						ResumeThread(thds[i]);
					}
					i++;
				}
				dwpThreadAffinityMask<<=1;
			}
			else
			{
				thds[i]=CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)StaticThreadpoolV,&MT_Thread[i],0,&tids[i]);
				ok=ok && (thds[i]!=NULL);
				i++;
			}
		}
		if (!ok)
		{
			FreeData();
			env->ThrowError("ResizeMT: Unable to create threads pool !");
		}
	}

  if (vi.IsRGB())
    subrange_top = vi.height - subrange_top - subrange_height; // why?

  // Create resampling program and pitch table
  resampling_program_luma  = func->GetResamplingProgram(vi.height, subrange_top, subrange_height, target_height, env);
  src_pitch_table_luma     = new int[vi.height];  
  if (src_pitch_table_luma==NULL)
  {
	  FreeData();
	  env->ThrowError("ResizeMT: Could not reserve memory in a resampler.");
  }
  
  resampler_luma_aligned   = GetResampler(env->GetCPUFlags(), true ,pixelsize, filter_storage_luma_aligned,   resampling_program_luma);
  resampler_luma_unaligned = GetResampler(env->GetCPUFlags(), false,pixelsize, filter_storage_luma_unaligned, resampling_program_luma);

  if (vi.IsPlanar() && !grey) {
    const int div   = 1 << shift_h;

    resampling_program_chroma = func->GetResamplingProgram(
                                  vi.height      >> shift_h,
                                  subrange_top    / div,
                                  subrange_height / div,
                                  target_height  >> shift_h,
                                  env);
    src_pitch_table_chromaU    = new int[vi.height >> shift_h];
    src_pitch_table_chromaV    = new int[vi.height >> shift_h];
	if ((src_pitch_table_chromaU==NULL) || (src_pitch_table_chromaV==NULL))
	{
		FreeData();
		env->ThrowError("ResizeMT: Could not reserve memory in a resampler.");
	}	
    resampler_chroma_aligned   = GetResampler(env->GetCPUFlags(), true ,pixelsize, filter_storage_chroma_aligned,   resampling_program_chroma);
    resampler_chroma_unaligned = GetResampler(env->GetCPUFlags(), false,pixelsize, filter_storage_chroma_unaligned, resampling_program_chroma);
  }

  // Change target video info size
  vi.height = target_height;
}


int __stdcall FilteredResizeV::SetCacheHints(int cachehints,int frame_range)
{
  switch (cachehints)
  {
  case CACHE_DONT_CACHE_ME:
    return 1;
  case CACHE_GET_MTMODE:
    return MT_MULTI_INSTANCE;
  default:
    return 0;
  }
}


uint8_t FilteredResizeV::CreateMTData(uint8_t max_threads,int32_t src_size_x,int32_t src_size_y,int32_t dst_size_x,int32_t dst_size_y,int UV_w,int UV_h)
{
	int32_t _y_min,_dh;

	if ((max_threads<=1) || (max_threads>threads_number))
	{
		MT_Data[0].top=true;
		MT_Data[0].bottom=true;
		MT_Data[0].src_Y_h_min=0;
		MT_Data[0].dst_Y_h_min=0;
		MT_Data[0].src_Y_h_max=src_size_y;
		MT_Data[0].dst_Y_h_max=dst_size_y;
		MT_Data[0].src_UV_h_min=0;
		MT_Data[0].dst_UV_h_min=0;
		if (UV_h>0)
		{
			MT_Data[0].src_UV_h_max=src_size_y >> UV_h;
			MT_Data[0].dst_UV_h_max=dst_size_y >> UV_h;
		}
		else
		{
			MT_Data[0].src_UV_h_max=src_size_y;
			MT_Data[0].dst_UV_h_max=dst_size_y;
		}
		MT_Data[0].src_Y_w=src_size_x;
		MT_Data[0].dst_Y_w=dst_size_x;
		if (UV_w>0)
		{
			MT_Data[0].src_UV_w=src_size_x >> UV_w;
			MT_Data[0].dst_UV_w=dst_size_x >> UV_w;
		}
		else
		{
			MT_Data[0].src_UV_w=src_size_x;
			MT_Data[0].dst_UV_w=dst_size_x;
		}
		return(1);
	}

	int32_t src_dh_Y,src_dh_UV,dst_dh_Y,dst_dh_UV;
	int32_t h_y;
	uint8_t i,max=0;

	dst_dh_Y=(dst_size_y+(uint32_t)max_threads-1)/(uint32_t)max_threads;
	if (dst_dh_Y<16) dst_dh_Y=16;
	if ((dst_dh_Y & 3)!=0) dst_dh_Y=((dst_dh_Y+3) >> 2) << 2;

	if (src_size_y==dst_size_y) src_dh_Y=dst_dh_Y;
	else
	{
		src_dh_Y=(src_size_y+(uint32_t)max_threads-1)/(uint32_t)max_threads;
		if (src_dh_Y<16) src_dh_Y=16;
		if ((src_dh_Y & 3)!=0) src_dh_Y=((src_dh_Y+3) >> 2) << 2;
	}

	if (src_size_y<dst_size_y)
	{
		_y_min=src_size_y;
		_dh=src_dh_Y;
	}
	else
	{
		_y_min=dst_size_y;
		_dh=dst_dh_Y;
	}
	h_y=0;
	while (h_y<(_y_min-16))
	{
		max++;
		h_y+=_dh;
	}

	if (max==1)
	{
		MT_Data[0].top=true;
		MT_Data[0].bottom=true;
		MT_Data[0].src_Y_h_min=0;
		MT_Data[0].dst_Y_h_min=0;
		MT_Data[0].src_Y_h_max=src_size_y;
		MT_Data[0].dst_Y_h_max=dst_size_y;
		MT_Data[0].src_UV_h_min=0;
		MT_Data[0].dst_UV_h_min=0;
		if (UV_h>0)
		{
			MT_Data[0].src_UV_h_max=src_size_y >> UV_h;
			MT_Data[0].dst_UV_h_max=dst_size_y >> UV_h;
		}
		else
		{
			MT_Data[0].src_UV_h_max=src_size_y;
			MT_Data[0].dst_UV_h_max=dst_size_y;
		}
		MT_Data[0].src_Y_w=src_size_x;
		MT_Data[0].dst_Y_w=dst_size_x;
		if (UV_w>0)
		{
			MT_Data[0].src_UV_w=src_size_x >> UV_w;
			MT_Data[0].dst_UV_w=dst_size_x >> UV_w;
		}
		else
		{
			MT_Data[0].src_UV_w=src_size_x;
			MT_Data[0].dst_UV_w=dst_size_x;
		}
		return(1);
	}

	src_dh_UV= (UV_h>0) ? src_dh_Y>>UV_h : src_dh_Y;
	dst_dh_UV= (UV_h>0) ? dst_dh_Y>>UV_h : dst_dh_Y;

	MT_Data[0].top=true;
	MT_Data[0].bottom=false;
	MT_Data[0].src_Y_h_min=0;
	MT_Data[0].src_Y_h_max=src_dh_Y;
	MT_Data[0].dst_Y_h_min=0;
	MT_Data[0].dst_Y_h_max=dst_dh_Y;
	MT_Data[0].src_UV_h_min=0;
	MT_Data[0].src_UV_h_max=src_dh_UV;
	MT_Data[0].dst_UV_h_min=0;
	MT_Data[0].dst_UV_h_max=dst_dh_UV;

	i=1;
	while (i<max)
	{
		MT_Data[i].top=false;
		MT_Data[i].bottom=false;
		MT_Data[i].src_Y_h_min=MT_Data[i-1].src_Y_h_max;
		MT_Data[i].src_Y_h_max=MT_Data[i].src_Y_h_min+src_dh_Y;
		MT_Data[i].dst_Y_h_min=MT_Data[i-1].dst_Y_h_max;
		MT_Data[i].dst_Y_h_max=MT_Data[i].dst_Y_h_min+dst_dh_Y;
		MT_Data[i].src_UV_h_min=MT_Data[i-1].src_UV_h_max;
		MT_Data[i].src_UV_h_max=MT_Data[i].src_UV_h_min+src_dh_UV;
		MT_Data[i].dst_UV_h_min=MT_Data[i-1].dst_UV_h_max;
		MT_Data[i].dst_UV_h_max=MT_Data[i].dst_UV_h_min+dst_dh_UV;
		i++;
	}

	MT_Data[max-1].bottom=true;
	MT_Data[max-1].src_Y_h_max=src_size_y;
	MT_Data[max-1].dst_Y_h_max=dst_size_y;
	if (UV_h>0)
	{
		MT_Data[max-1].src_UV_h_max=src_size_y >> UV_h;
		MT_Data[max-1].dst_UV_h_max=dst_size_y >> UV_h;
	}
	else
	{
		MT_Data[max-1].src_UV_h_max=src_size_y;
		MT_Data[max-1].dst_UV_h_max=dst_size_y;
	}

	for (i=0; i<max; i++)
	{
		MT_Data[i].src_Y_w=src_size_x;
		MT_Data[i].dst_Y_w=dst_size_x;
		if (UV_w>0)
		{
			MT_Data[i].src_UV_w=src_size_x >> UV_w;
			MT_Data[i].dst_UV_w=dst_size_x >> UV_w;
		}
		else
		{
			MT_Data[i].src_UV_w=src_size_x;
			MT_Data[i].dst_UV_w=dst_size_x;
		}
	}

	return(max);
}

void FilteredResizeV::ResamplerLumaAlignedMT(uint8_t thread_num)
{
	const MT_Data_Info mt_data_inf=MT_Data[thread_num];

	resampler_luma_aligned(mt_data_inf.dst1,mt_data_inf.src1,mt_data_inf.dst_pitch1,mt_data_inf.src_pitch1,
		mt_data_inf.resampling_program_luma,mt_data_inf.src_Y_w,mt_data_inf.dst_Y_h_min,mt_data_inf.dst_Y_h_max,
		mt_data_inf.src_pitch_table_luma,mt_data_inf.filter_storage_luma);
}


void FilteredResizeV::ResamplerLumaUnalignedMT(uint8_t thread_num)
{
	const MT_Data_Info mt_data_inf=MT_Data[thread_num];

	resampler_luma_unaligned(mt_data_inf.dst1,mt_data_inf.src1,mt_data_inf.dst_pitch1,mt_data_inf.src_pitch1,
		mt_data_inf.resampling_program_luma,mt_data_inf.src_Y_w,mt_data_inf.dst_Y_h_min,mt_data_inf.dst_Y_h_max,
		mt_data_inf.src_pitch_table_luma,mt_data_inf.filter_storage_luma);
}


void FilteredResizeV::ResamplerUChromaAlignedMT(uint8_t thread_num)
{
	const MT_Data_Info mt_data_inf=MT_Data[thread_num];

	resampler_chroma_aligned(mt_data_inf.dst2,mt_data_inf.src2,mt_data_inf.dst_pitch2,mt_data_inf.src_pitch2,
		mt_data_inf.resampling_program_chroma,mt_data_inf.src_UV_w,mt_data_inf.dst_UV_h_min,mt_data_inf.dst_UV_h_max,
		mt_data_inf.src_pitch_table_chromaU,mt_data_inf.filter_storage_chromaU);
}



void FilteredResizeV::ResamplerUChromaUnalignedMT(uint8_t thread_num)
{
	const MT_Data_Info mt_data_inf=MT_Data[thread_num];

	resampler_chroma_unaligned(mt_data_inf.dst2,mt_data_inf.src2,mt_data_inf.dst_pitch2,mt_data_inf.src_pitch2,
		mt_data_inf.resampling_program_chroma,mt_data_inf.src_UV_w,mt_data_inf.dst_UV_h_min,mt_data_inf.dst_UV_h_max,
		mt_data_inf.src_pitch_table_chromaU,mt_data_inf.filter_storage_chromaU);
}


void FilteredResizeV::ResamplerVChromaAlignedMT(uint8_t thread_num)
{
	const MT_Data_Info mt_data_inf=MT_Data[thread_num];

	resampler_chroma_aligned(mt_data_inf.dst3,mt_data_inf.src3,mt_data_inf.dst_pitch3,mt_data_inf.src_pitch3,
		mt_data_inf.resampling_program_chroma,mt_data_inf.src_UV_w,mt_data_inf.dst_UV_h_min,mt_data_inf.dst_UV_h_max,
		mt_data_inf.src_pitch_table_chromaV,mt_data_inf.filter_storage_chromaV);
}


void FilteredResizeV::ResamplerVChromaUnalignedMT(uint8_t thread_num)
{
	const MT_Data_Info mt_data_inf=MT_Data[thread_num];

	resampler_chroma_unaligned(mt_data_inf.dst3,mt_data_inf.src3,mt_data_inf.dst_pitch3,mt_data_inf.src_pitch3,
		mt_data_inf.resampling_program_chroma,mt_data_inf.src_UV_w,mt_data_inf.dst_UV_h_min,mt_data_inf.dst_UV_h_max,
		mt_data_inf.src_pitch_table_chromaV,mt_data_inf.filter_storage_chromaV);
}


DWORD WINAPI FilteredResizeV::StaticThreadpoolV( LPVOID lpParam )
{
	const MT_Data_Thread *data=(const MT_Data_Thread *)lpParam;
	FilteredResizeV *ptrClass=(FilteredResizeV *)data->pClass;
	
	while (true)
	{
		WaitForSingleObject(data->nextJob,INFINITE);
		switch(data->f_process)
		{
			case 1 : ptrClass->ResamplerLumaAlignedMT(data->thread_Id);
				break;
			case 2 : ptrClass->ResamplerLumaUnalignedMT(data->thread_Id);
				break;
			case 3 : ptrClass->ResamplerUChromaAlignedMT(data->thread_Id);
				break;
			case 4 : ptrClass->ResamplerUChromaUnalignedMT(data->thread_Id);
				break;
			case 5 : ptrClass->ResamplerVChromaAlignedMT(data->thread_Id);
				break;
			case 6 : ptrClass->ResamplerVChromaUnalignedMT(data->thread_Id);
				break;
			case 255 : return(0); break;
			default : ;
		}
		ResetEvent(data->nextJob);
		SetEvent(data->jobFinished);
	}
}


PVideoFrame __stdcall FilteredResizeV::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrame(vi);
  const int src_pitch_Y = src->GetPitch();
  const int dst_pitch_Y = dst->GetPitch();
  const BYTE* srcp_Y = src->GetReadPtr();
        BYTE* dstp_Y = dst->GetWritePtr();

	const int src_pitch_U = (!grey && vi.IsPlanar()) ? src->GetPitch(PLANAR_U) : 0;
	const int dst_pitch_U = (!grey && vi.IsPlanar()) ? dst->GetPitch(PLANAR_U) : 0;
	const BYTE* srcp_U = (!grey && vi.IsPlanar()) ? src->GetReadPtr(PLANAR_U) : NULL;
	BYTE* dstp_U = (!grey && vi.IsPlanar()) ? dst->GetWritePtr(PLANAR_U) : NULL;

	const int src_pitch_V = (!grey && vi.IsPlanar()) ? src->GetPitch(PLANAR_V) : 0;
	const int dst_pitch_V = (!grey && vi.IsPlanar()) ? dst->GetPitch(PLANAR_V) : 0;
	const BYTE* srcp_V = (!grey && vi.IsPlanar()) ? src->GetReadPtr(PLANAR_V) : NULL;
	BYTE* dstp_V = (!grey && vi.IsPlanar()) ? dst->GetWritePtr(PLANAR_V) : NULL;


  WaitForSingleObject(ghMutex,INFINITE);

  // Create pitch table
  if (src_pitch_luma != src->GetPitch()) {
    src_pitch_luma = src->GetPitch();
    resize_v_create_pitch_table(src_pitch_table_luma, src_pitch_luma, src->GetHeight(),pixelsize);
  }

  if ((!grey && vi.IsPlanar()) && (src_pitch_chromaU != src->GetPitch(PLANAR_U))) {
    src_pitch_chromaU = src->GetPitch(PLANAR_U);
    resize_v_create_pitch_table(src_pitch_table_chromaU, src_pitch_chromaU, src->GetHeight(PLANAR_U),pixelsize);
  }

  if ((!grey && vi.IsPlanar()) && (src_pitch_chromaV != src->GetPitch(PLANAR_V))) {
    src_pitch_chromaV = src->GetPitch(PLANAR_V);
    resize_v_create_pitch_table(src_pitch_table_chromaV, src_pitch_chromaV, src->GetHeight(PLANAR_V),pixelsize);
  }

	for(uint8_t i=0; i<threads_number; i++)
	{
		MT_Data[i].src1=srcp_Y;
		MT_Data[i].src2=srcp_U;
		MT_Data[i].src3=srcp_V;
		MT_Data[i].src_pitch1=src_pitch_Y;
		MT_Data[i].src_pitch2=src_pitch_U;
		MT_Data[i].src_pitch3=src_pitch_V;
		MT_Data[i].dst1=dstp_Y+(MT_Data[i].dst_Y_h_min*dst_pitch_Y);
		MT_Data[i].dst2=dstp_U+(MT_Data[i].dst_UV_h_min*dst_pitch_U);
		MT_Data[i].dst3=dstp_V+(MT_Data[i].dst_UV_h_min*dst_pitch_V);
		MT_Data[i].dst_pitch1=dst_pitch_Y;
		MT_Data[i].dst_pitch2=dst_pitch_U;
		MT_Data[i].dst_pitch3=dst_pitch_V;
		if (IsPtrAligned(srcp_Y, 16) && (src_pitch_Y & 15) == 0)
			MT_Data[i].filter_storage_luma=filter_storage_luma_aligned;
		else
			MT_Data[i].filter_storage_luma=filter_storage_luma_unaligned;
		MT_Data[i].src_pitch_table_luma=src_pitch_table_luma;
		MT_Data[i].src_pitch_table_chromaU=src_pitch_table_chromaU;
		MT_Data[i].src_pitch_table_chromaV=src_pitch_table_chromaV;
		MT_Data[i].resampling_program_luma=resampling_program_luma;
		MT_Data[i].resampling_program_chroma=resampling_program_chroma;
		if (IsPtrAligned(srcp_U, 16) && (src_pitch_U & 15) == 0)
			MT_Data[i].filter_storage_chromaU=filter_storage_chroma_aligned;
		else
			MT_Data[i].filter_storage_chromaU=filter_storage_chroma_unaligned;
		if (IsPtrAligned(srcp_V, 16) && (src_pitch_V & 15) == 0)
			MT_Data[i].filter_storage_chromaV=filter_storage_chroma_aligned;
		else
			MT_Data[i].filter_storage_chromaV=filter_storage_chroma_unaligned;
	}


	if (threads_number>1)
	{
		uint8_t f_proc;

		if (IsPtrAligned(srcp_Y, 16) && (src_pitch_Y & 15) == 0) f_proc=1;
		else f_proc=2;

		for(uint8_t i=0; i<threads_number; i++)
		{
			MT_Thread[i].f_process=f_proc;
			ResetEvent(MT_Thread[i].jobFinished);
			SetEvent(MT_Thread[i].nextJob);
		}
		for(uint8_t i=0; i<threads_number; i++)
			WaitForSingleObject(MT_Thread[i].jobFinished,INFINITE);

		if (!vi.IsY8() && vi.IsPlanar())
		{
			if (IsPtrAligned(srcp_U, 16) && (src_pitch_U & 15) == 0) f_proc=3;
			else f_proc=4;

			for(uint8_t i=0; i<threads_number; i++)
			{
				MT_Thread[i].f_process=f_proc;
				ResetEvent(MT_Thread[i].jobFinished);
				SetEvent(MT_Thread[i].nextJob);
			}
			for(uint8_t i=0; i<threads_number; i++)
				WaitForSingleObject(MT_Thread[i].jobFinished,INFINITE);

			if (IsPtrAligned(srcp_V, 16) && (src_pitch_V & 15) == 0) f_proc=5;
			else f_proc=6;

			for(uint8_t i=0; i<threads_number; i++)
			{
				MT_Thread[i].f_process=f_proc;
				ResetEvent(MT_Thread[i].jobFinished);
				SetEvent(MT_Thread[i].nextJob);
			}
			for(uint8_t i=0; i<threads_number; i++)
				WaitForSingleObject(MT_Thread[i].jobFinished,INFINITE);
		}

		for(uint8_t i=0; i<threads_number; i++)
			MT_Thread[i].f_process=0;
	}
	else
	{
		// Do resizing
		if (IsPtrAligned(srcp_Y, 16) && (src_pitch_Y & 15) == 0)
			ResamplerLumaAlignedMT(0);
		else
			ResamplerLumaUnalignedMT(0);
    
		if (!vi.IsY8() && vi.IsPlanar())
		{
			// Plane U resizing   
			if (IsPtrAligned(srcp_U, 16) && (src_pitch_U & 15) == 0)
				ResamplerUChromaAlignedMT(0);
			else
				ResamplerUChromaUnalignedMT(0);

			// Plane V resizing
			if (IsPtrAligned(srcp_V, 16) && (src_pitch_V & 15) == 0)
				ResamplerVChromaAlignedMT(0);
			else
				ResamplerVChromaUnalignedMT(0);
		}
	}

	ReleaseMutex(ghMutex);

  return dst;
}

ResamplerV FilteredResizeV::GetResampler(int CPU, bool aligned,int pixelsize, void*& storage, ResamplingProgram* program)
{
  if (program->filter_size == 1) {
    // Fast pointresize
    switch (pixelsize) // AVS16
    {
    case 1: return resize_v_planar_pointresize<uint8_t>;
    case 2: return resize_v_planar_pointresize<uint16_t>;
    default: // case 4:
      return resize_v_planar_pointresize<float>;
    }
  }
  else {
    // Other resizers
    if (pixelsize == 1)
    {
      if (CPU & CPUF_SSSE3) {
        if (aligned && CPU & CPUF_SSE4_1) {
          return resize_v_ssse3_planar<simd_load_streaming>;
        }
        else if (aligned) { // SSSE3 aligned
          return resize_v_ssse3_planar<simd_load_aligned>;
        }
        else if (CPU & CPUF_SSE3) { // SSE3 lddqu
          return resize_v_ssse3_planar<simd_load_unaligned_sse3>;
        }
        else { // SSSE3 unaligned
          return resize_v_ssse3_planar<simd_load_unaligned>;
        }
      }
      else if (CPU & CPUF_SSE2) {
        if (aligned && CPU & CPUF_SSE4_1) { // SSE4.1 movntdqa constantly provide ~2% performance increase in my testing
          return resize_v_sse2_planar<simd_load_streaming>;
        }
        else if (aligned) { // SSE2 aligned
          return resize_v_sse2_planar<simd_load_aligned>;
        }
        else if (CPU & CPUF_SSE3) { // SSE2 lddqu
          return resize_v_sse2_planar<simd_load_unaligned_sse3>;
        }
        else { // SSE2 unaligned
          return resize_v_sse2_planar<simd_load_unaligned>;
        }
#ifdef X86_32
      }
      else if (CPU & CPUF_MMX) {
        return resize_v_mmx_planar;
#endif
      }
      else { // C version
        return resize_v_c_planar;
      }
    } // todo: sse
    else if (pixelsize == 2) {
      return resize_v_c_planar_s;
    }
    else { // if (pixelsize== 4) 
      return resize_v_c_planar_f;
    }
  }	
}


void FilteredResizeV::FreeData(void) 
{
	int16_t i;

  if (resampling_program_luma!=NULL)   { delete resampling_program_luma; }
  if (resampling_program_chroma!=NULL) { delete resampling_program_chroma; }
  if (src_pitch_table_luma!=NULL)    { delete[] src_pitch_table_luma; }
  if (src_pitch_table_chromaU!=NULL) { delete[] src_pitch_table_chromaU; }
  if (src_pitch_table_chromaV!=NULL) { delete[] src_pitch_table_chromaV; }

  myalignedfree(filter_storage_luma_aligned);
  myalignedfree(filter_storage_luma_unaligned);
  myalignedfree(filter_storage_chroma_aligned);
  myalignedfree(filter_storage_chroma_unaligned);

	if (threads_number>1)
	{
		for (i=threads_number-1; i>=0; i--)
		{
			if (thds[i]!=NULL)
			{
				MT_Thread[i].f_process=255;
				SetEvent(MT_Thread[i].nextJob);
				WaitForSingleObject(thds[i],INFINITE);
				myCloseHandle(thds[i]);
			}
		}
		for (i=threads_number-1; i>=0; i--)
		{
			myCloseHandle(MT_Thread[i].nextJob);
			myCloseHandle(MT_Thread[i].jobFinished);
		}
	}
	myCloseHandle(ghMutex);
}


FilteredResizeV::~FilteredResizeV(void)
{
	FreeData();
}


/**********************************************
 *******   Resampling Factory Methods   *******
 **********************************************/


PClip FilteredResizeMT::CreateResizeV(PClip clip, double subrange_top, double subrange_height, int target_height,
                    int _threads,bool _avsp,ResamplingFunction* func, IScriptEnvironment* env)
{
  return new FilteredResizeV(clip, subrange_top, subrange_height, target_height,_threads,_avsp,func, env);
}


PClip FilteredResizeMT::CreateResizeH(PClip clip, double subrange_top, double subrange_height, int target_height,
                    int _threads,bool _avsp,ResamplingFunction* func, IScriptEnvironment* env)
{
  return new FilteredResizeH(clip, subrange_top, subrange_height, target_height,_threads,_avsp,func, env);
}


PClip FilteredResizeMT::CreateResize(PClip clip, int target_width, int target_height,int _threads,
	const AVSValue* args,ResamplingFunction* f, IScriptEnvironment* env)
{
  const VideoInfo& vi = clip->GetVideoInfo();
  const double subrange_left = args[0].AsFloat(0), subrange_top = args[1].AsFloat(0);
  
  if (target_height <= 0)
    env->ThrowError("ResizeMT: Height must be greater than 0.");

  if (target_width <= 0) {
    env->ThrowError("ResizeMT: Width must be greater than 0.");
  }

  const bool avsp=env->FunctionExists("ConvertToFloat");

  const bool grey = (avsp) ? vi.IsY8() || vi.IsColorSpace(VideoInfo::CS_Y16) || vi.IsColorSpace(VideoInfo::CS_Y32) : vi.IsY8();

  bool fast_resize = ((env->GetCPUFlags() & CPUF_SSSE3) == CPUF_SSSE3 ) && vi.IsPlanar() && ((target_width & 3) == 0);  
  if (fast_resize && vi.IsYUV() && !grey)
  {
    const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int dst_chroma_width = target_width >> shift;

    if ((dst_chroma_width & 3) != 0) fast_resize = false;
  }  

  if (vi.IsPlanar() && !grey)
  {
    int  mask;
	
	mask = (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1;
    if (target_height & mask)
      env->ThrowError("ResizeMT: Planar destination height must be a multiple of %d.", mask+1);
  
    mask = (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1;
    if (target_width & mask)
      env->ThrowError("ResizeMT: Planar destination height must be a multiple of %d.", mask+1);
  
  }

  
	if ((_threads<0) || (_threads>MAX_MT_THREADS))
	{
		char buffer_in[1024];

		sprintf_s(buffer_in,1024,"ResizeMT : [threads] must be between 0 and %ld.",MAX_MT_THREADS);
		env->ThrowError(buffer_in);
	}

  double subrange_width = args[2].AsDblDef(vi.width), subrange_height = args[3].AsDblDef(vi.height);
  // Crop style syntax
  if (subrange_width  <= 0.0) subrange_width  = vi.width  - subrange_left + subrange_width;
  if (subrange_height <= 0.0) subrange_height = vi.height - subrange_top  + subrange_height;

  PClip result;
  // ensure that the intermediate area is maximal
  const double area_FirstH = subrange_height * target_width;
  const double area_FirstV = subrange_width * target_height;

  const bool FTurnL=(env->FunctionExists("FTurnLeft") && ((env->GetCPUFlags() & CPUF_SSE2)!=0)) && (!vi.IsRGB());
  const bool FTurnR=(env->FunctionExists("FTurnRight") && ((env->GetCPUFlags() & CPUF_SSE2)!=0)) && (!vi.IsRGB());

  auto turnRightFunction = (FTurnR) ? "FTurnRight" : "TurnRight";
  auto turnLeftFunction =  (FTurnL) ? "FTurnLeft" : "TurnLeft";

  if (!fast_resize)
  {
	  if (area_FirstH < area_FirstV)
	  {
		if ((subrange_top==0) && (subrange_height==target_height) && (subrange_height==vi.height))
			result=clip;
		else
		{
			if ((subrange_top==int(subrange_top)) && (subrange_height==target_height)
			   && (subrange_top>=0) && ((subrange_top+subrange_height)<= vi.height))
			{
				const int mask = (vi.IsYUV() && !vi.IsY8() && !vi.IsColorSpace(VideoInfo::CS_Y16) && !vi.IsColorSpace(VideoInfo::CS_Y32)) ? (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1 : 0;

				if (((int(subrange_top) | int(subrange_height)) & mask) == 0)
				{
					  AVSValue sargs[6] = {clip,0,int(subrange_top),vi.width,int(subrange_height),0};
					  result=env->Invoke("Crop",AVSValue(sargs,6)).AsClip();
				}
				else result = CreateResizeV(clip, subrange_top, subrange_height, target_height,_threads, avsp, f, env);
			}
			else result = CreateResizeV(clip, subrange_top, subrange_height, target_height,_threads,avsp, f, env);
		}
		if (!((subrange_left==0) && (subrange_width==target_width) && (subrange_width==vi.width)))
		{
			if ((subrange_left==int(subrange_left)) && (subrange_width==target_width)
				&& (subrange_left>=0) && ((subrange_left+subrange_width)<=vi.width))
			{
				const int mask = (vi.IsYUV() && !vi.IsY8() && !vi.IsColorSpace(VideoInfo::CS_Y16) && !vi.IsColorSpace(VideoInfo::CS_Y32)) ? (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1 : 0;

			    if (((int(subrange_left) | int(subrange_width)) & mask) == 0)
				{
				  AVSValue sargs[6] = {result,int(subrange_left),0,int(subrange_width),vi.height,0};
				  result=env->Invoke("Crop",AVSValue(sargs,6)).AsClip();
				}
				else
				{
					if (!vi.IsRGB())
					{
						if (vi.IsYV16() || vi.IsYUY2() || vi.IsYV411())
						{
							const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
							const int div   = 1 << shift;

							AVSValue v,vv,vu;

							vu = env->Invoke("UtoY8",result).AsClip();
							vv = env->Invoke("VtoY8",result).AsClip();
							v = env->Invoke("ConvertToY8",result).AsClip();
							v = env->Invoke(turnRightFunction,v).AsClip();
							vu = env->Invoke(turnRightFunction,vu).AsClip();
							vv = env->Invoke(turnRightFunction,vv).AsClip();
							v = CreateResizeV(v.AsClip(), subrange_left, subrange_width, target_width,_threads,avsp, f, env);
							vu = CreateResizeV(vu.AsClip(), subrange_left/div, subrange_width/div, target_width >> shift,_threads,avsp, f, env);
							vv = CreateResizeV(vv.AsClip(), subrange_left/div, subrange_width/div, target_width >> shift,_threads,avsp, f, env);
							v = env->Invoke(turnLeftFunction,v).AsClip();
							vu = env->Invoke(turnLeftFunction,vu).AsClip();
							vv = env->Invoke(turnLeftFunction,vv).AsClip();

						    AVSValue ytouvargs[3] = {vu,vv,v};
						    result=env->Invoke("YtoUV",AVSValue(ytouvargs,3)).AsClip();
						    if (vi.IsYUY2()) result=env->Invoke("ConvertToYUY2",result).AsClip();
						}
						else
						{
							result=env->Invoke(turnRightFunction,result).AsClip();
							result=CreateResizeV(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
							result=env->Invoke(turnLeftFunction,result).AsClip();
						}
					}
					else
					{
						result=env->Invoke(turnRightFunction,result).AsClip();
						result=CreateResizeV(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
						result=env->Invoke(turnLeftFunction,result).AsClip();
					}
				}
			}
			else
			{
				if (!vi.IsRGB())
				{
				    if (vi.IsYV16() || vi.IsYUY2() || vi.IsYV411())
					{
						const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
						const int div   = 1 << shift;

						AVSValue v,vv,vu;

						vu = env->Invoke("UtoY8",result).AsClip();
						vv = env->Invoke("VtoY8",result).AsClip();
						v = env->Invoke("ConvertToY8",result).AsClip();
						v = env->Invoke(turnRightFunction,v).AsClip();
						vu = env->Invoke(turnRightFunction,vu).AsClip();
						vv = env->Invoke(turnRightFunction,vv).AsClip();
						v = CreateResizeV(v.AsClip(), subrange_left, subrange_width, target_width,_threads,avsp, f, env);
						vu = CreateResizeV(vu.AsClip(), subrange_left/div, subrange_width/div, target_width >> shift,_threads,avsp, f, env);
						vv = CreateResizeV(vv.AsClip(), subrange_left/div, subrange_width/div, target_width >> shift,_threads,avsp, f, env);
						v = env->Invoke(turnLeftFunction,v).AsClip();
						vu = env->Invoke(turnLeftFunction,vu).AsClip();
						vv = env->Invoke(turnLeftFunction,vv).AsClip();
							
					    AVSValue ytouvargs[3] = {vu,vv,v};
					    result=env->Invoke("YtoUV",AVSValue(ytouvargs,3)).AsClip();
					    if (vi.IsYUY2()) result=env->Invoke("ConvertToYUY2",result).AsClip();
					}
					else
					{
						result=env->Invoke(turnRightFunction,result).AsClip();
						result=CreateResizeV(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
						result=env->Invoke(turnLeftFunction,result).AsClip();
					}
				}
				else
				{
					result=env->Invoke(turnRightFunction,result).AsClip();
					result=CreateResizeV(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
					result=env->Invoke(turnLeftFunction,result).AsClip();
				}
			}
		}
	  }
	  else
	  {
		if ((subrange_left==0) && (subrange_width==target_width) && (subrange_width==vi.width))
		{
			result=clip;
		}
		else
		{
			if ((subrange_left==int(subrange_left)) && (subrange_width==target_width)
				&& (subrange_left>=0) && ((subrange_left+subrange_width)<=vi.width))
			{
				const int mask = (vi.IsYUV() && !vi.IsY8() && !vi.IsColorSpace(VideoInfo::CS_Y16) && !vi.IsColorSpace(VideoInfo::CS_Y32)) ? (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1 : 0;

			    if (((int(subrange_left) | int(subrange_width)) & mask) == 0)
				{
				  AVSValue sargs[6] = {clip,int(subrange_left),0,int(subrange_width),vi.height,0};
				  result=env->Invoke("Crop",AVSValue(sargs,6)).AsClip();
				}
				else
				{
					if (!vi.IsRGB())
					{
					    if (vi.IsYV16() || vi.IsYUY2() || vi.IsYV411())
						{
							const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
							const int div   = 1 << shift;

							AVSValue v,vv,vu;

							vu = env->Invoke("UtoY8",clip).AsClip();
							vv = env->Invoke("VtoY8",clip).AsClip();
							v = env->Invoke("ConvertToY8",clip).AsClip();
							v = env->Invoke(turnRightFunction,v).AsClip();
							vu = env->Invoke(turnRightFunction,vu).AsClip();
							vv = env->Invoke(turnRightFunction,vv).AsClip();
							v = CreateResizeV(v.AsClip(), subrange_left, subrange_width, target_width,_threads,avsp, f, env);
							vu = CreateResizeV(vu.AsClip(), subrange_left/div, subrange_width/div, target_width >> shift,_threads,avsp, f, env);
							vv = CreateResizeV(vv.AsClip(), subrange_left/div, subrange_width/div, target_width >> shift,_threads,avsp, f, env);
							v = env->Invoke(turnLeftFunction,v).AsClip();
							vu = env->Invoke(turnLeftFunction,vu).AsClip();
							vv = env->Invoke(turnLeftFunction,vv).AsClip();

						    AVSValue ytouvargs[3] = {vu,vv,v};
						    result=env->Invoke("YtoUV",AVSValue(ytouvargs,3)).AsClip();
						    if (vi.IsYUY2()) result=env->Invoke("ConvertToYUY2",result).AsClip();
						}
						else
						{
							result=env->Invoke(turnRightFunction,clip).AsClip();
							result=CreateResizeV(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
							result=env->Invoke(turnLeftFunction,result).AsClip();
						}
					}
					else
					{
						result=env->Invoke(turnRightFunction,clip).AsClip();
						result=CreateResizeV(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
						result=env->Invoke(turnLeftFunction,result).AsClip();
					}
				}
			}
			else
			{
				if (!vi.IsRGB())
				{
					if (vi.IsYV16() || vi.IsYUY2() || vi.IsYV411())
					{
						const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
						const int div   = 1 << shift;

						AVSValue v,vv,vu;

						vu = env->Invoke("UtoY8",clip).AsClip();
						vv = env->Invoke("VtoY8",clip).AsClip();
						v = env->Invoke("ConvertToY8",clip).AsClip();
						v = env->Invoke(turnRightFunction,v).AsClip();
						vu = env->Invoke(turnRightFunction,vu).AsClip();
						vv = env->Invoke(turnRightFunction,vv).AsClip();
						v = CreateResizeV(v.AsClip(), subrange_left, subrange_width, target_width,_threads,avsp, f, env);
						vu = CreateResizeV(vu.AsClip(), subrange_left/div, subrange_width/div, target_width >> shift,_threads,avsp, f, env);
						vv = CreateResizeV(vv.AsClip(), subrange_left/div, subrange_width/div, target_width >> shift,_threads,avsp, f, env);
						v = env->Invoke(turnLeftFunction,v).AsClip();
						vu = env->Invoke(turnLeftFunction,vu).AsClip();
						vv = env->Invoke(turnLeftFunction,vv).AsClip();

					    AVSValue ytouvargs[3] = {vu,vv,v};
					    result=env->Invoke("YtoUV",AVSValue(ytouvargs,3)).AsClip();
					    if (vi.IsYUY2()) result=env->Invoke("ConvertToYUY2",result).AsClip();
					}
					else
					{
						result=env->Invoke(turnRightFunction,clip).AsClip();
						result=CreateResizeV(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
						result=env->Invoke(turnLeftFunction,result).AsClip();
					}
				}
				else
				{
					result=env->Invoke(turnRightFunction,clip).AsClip();
					result=CreateResizeV(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
					result=env->Invoke(turnLeftFunction,result).AsClip();
				}
			}
		}
		if (!((subrange_top==0) && (subrange_height==target_height) && (subrange_height==vi.height)))
		{
			if ((subrange_top==int(subrange_top)) && (subrange_height==target_height)
			&& (subrange_top>=0) && ((subrange_top+subrange_height)<= vi.height))
			{
				const int mask = (vi.IsYUV() && !vi.IsY8() && !vi.IsColorSpace(VideoInfo::CS_Y16) && !vi.IsColorSpace(VideoInfo::CS_Y32)) ? (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1 : 0;
				
				if (((int(subrange_top) | int(subrange_height)) & mask) == 0)
				{
					  AVSValue sargs[6] = {result,0,int(subrange_top),vi.width,int(subrange_height),0};
					  result=env->Invoke("Crop",AVSValue(sargs,6)).AsClip();
				}
				else result = CreateResizeV(result, subrange_top, subrange_height, target_height,_threads,avsp, f, env);
			}
			else result = CreateResizeV(result, subrange_top, subrange_height, target_height,_threads,avsp, f, env);
		}
	  }
  }
  else
  {
	  if (area_FirstH < area_FirstV)
	  {
		if ((subrange_top==0) && (subrange_height==target_height) && (subrange_height==vi.height))
			result=clip;
		else
		{
			if ((subrange_top==int(subrange_top)) && (subrange_height==target_height)
			   && (subrange_top>=0) && ((subrange_top+subrange_height)<= vi.height))
			{
				const int mask = (vi.IsYUV() && !vi.IsY8() && !vi.IsColorSpace(VideoInfo::CS_Y16) && !vi.IsColorSpace(VideoInfo::CS_Y32)) ? (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1 : 0;

				if (((int(subrange_top) | int(subrange_height)) & mask) == 0)
				{
					  AVSValue sargs[6] = {clip,0,int(subrange_top),vi.width,int(subrange_height),0};
					  result=env->Invoke("Crop",AVSValue(sargs,6)).AsClip();
				}
				else result = CreateResizeV(clip, subrange_top, subrange_height, target_height,_threads, avsp, f, env);
			}
			else result = CreateResizeV(clip, subrange_top, subrange_height, target_height,_threads,avsp, f, env);
		}
		if (!((subrange_left==0) && (subrange_width==target_width) && (subrange_width==vi.width)))
		{
			if ((subrange_left==int(subrange_left)) && (subrange_width==target_width)
				&& (subrange_left>=0) && ((subrange_left+subrange_width)<=vi.width))
			{
				const int mask = (vi.IsYUV() && !vi.IsY8() && !vi.IsColorSpace(VideoInfo::CS_Y16) && !vi.IsColorSpace(VideoInfo::CS_Y32)) ? (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1 : 0;

			    if (((int(subrange_left) | int(subrange_width)) & mask) == 0)
				{
				  AVSValue sargs[6] = {result,int(subrange_left),0,int(subrange_width),vi.height,0};
				  result=env->Invoke("Crop",AVSValue(sargs,6)).AsClip();
				}
				else result = CreateResizeH(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
			}
			else result = CreateResizeH(result, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
		}
	  }
	  else
	  {
		if ((subrange_left==0) && (subrange_width==target_width) && (subrange_width==vi.width))
			result=clip;
		else
		{
			if ((subrange_left==int(subrange_left)) && (subrange_width==target_width)
				&& (subrange_left>=0) && ((subrange_left+subrange_width)<=vi.width))
			{
				const int mask = (vi.IsYUV() && !vi.IsY8() && !vi.IsColorSpace(VideoInfo::CS_Y16) && !vi.IsColorSpace(VideoInfo::CS_Y32)) ? (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1 : 0;

			    if (((int(subrange_left) | int(subrange_width)) & mask) == 0)
				{
				  AVSValue sargs[6] = {clip,int(subrange_left),0,int(subrange_width),vi.height,0};
				  result=env->Invoke("Crop",AVSValue(sargs,6)).AsClip();
				}
				else result = CreateResizeH(clip, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
			}
			else result = CreateResizeH(clip, subrange_left, subrange_width, target_width,_threads,avsp, f, env);
		}

		if (!((subrange_top==0) && (subrange_height==target_height) && (subrange_height==vi.height)))
		{
			if ((subrange_top==int(subrange_top)) && (subrange_height==target_height)
			&& (subrange_top>=0) && ((subrange_top+subrange_height)<= vi.height))
			{
				const int mask = (vi.IsYUV() && !vi.IsY8() && !vi.IsColorSpace(VideoInfo::CS_Y16) && !vi.IsColorSpace(VideoInfo::CS_Y32)) ? (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1 : 0;
				
				if (((int(subrange_top) | int(subrange_height)) & mask) == 0)
				{
					  AVSValue sargs[6] = {result,0,int(subrange_top),vi.width,int(subrange_height),0};
					  result=env->Invoke("Crop",AVSValue(sargs,6)).AsClip();
				}
				else result = CreateResizeV(result, subrange_top, subrange_height, target_height,_threads,avsp, f, env);
			}
			else result = CreateResizeV(result, subrange_top, subrange_height, target_height,_threads,avsp, f, env);
		}
	  }
  }

  return result;
}

AVSValue __cdecl FilteredResizeMT::Create_PointResize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[7].AsInt(0), &args[3],
                       &PointFilter(), env );
}


AVSValue __cdecl FilteredResizeMT::Create_BilinearResize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[7].AsInt(0), &args[3],
                       &TriangleFilter(), env );
}


AVSValue __cdecl FilteredResizeMT::Create_BicubicResize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[9].AsInt(0), &args[5],
                       &MitchellNetravaliFilter(args[3].AsDblDef(1./3.), args[4].AsDblDef(1./3.)), env );
}

AVSValue __cdecl FilteredResizeMT::Create_LanczosResize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[8].AsInt(0), &args[3],
                       &LanczosFilter(args[7].AsInt(3)), env );
}

AVSValue __cdecl FilteredResizeMT::Create_Lanczos4Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[7].AsInt(0), &args[3],
                       &LanczosFilter(4), env );
}

AVSValue __cdecl FilteredResizeMT::Create_BlackmanResize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[8].AsInt(0), &args[3],
                       &BlackmanFilter(args[7].AsInt(4)), env );
}

AVSValue __cdecl FilteredResizeMT::Create_Spline16Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[7].AsInt(0), &args[3],
                       &Spline16Filter(), env );
}

AVSValue __cdecl FilteredResizeMT::Create_Spline36Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[7].AsInt(0), &args[3],
                       &Spline36Filter(), env );
}

AVSValue __cdecl FilteredResizeMT::Create_Spline64Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[7].AsInt(0), &args[3],
                       &Spline64Filter(), env );
}

AVSValue __cdecl FilteredResizeMT::Create_GaussianResize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[8].AsInt(0), &args[3],
                       &GaussianFilter(args[7].AsFloat(30.0f)), env );
}

AVSValue __cdecl FilteredResizeMT::Create_SincResize(AVSValue args, void*, IScriptEnvironment* env)
{
  return CreateResize( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),args[8].AsInt(0), &args[3],
                       &SincFilter(args[7].AsInt(4)), env );
}


const AVS_Linkage *AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
	AVS_linkage = vectors;

	env->AddFunction("PointResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[threads]i",
		FilteredResizeMT::Create_PointResize, 0);
	env->AddFunction("BilinearResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[threads]i",
		FilteredResizeMT::Create_BilinearResize, 0);
	env->AddFunction("BicubicResizeMT", "c[target_width]i[target_height]i[b]f[c]f[src_left]f[src_top]f[src_width]f[src_height]f[threads]i",
		FilteredResizeMT::Create_BicubicResize, 0);
	env->AddFunction("LanczosResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[taps]i[threads]i",
		FilteredResizeMT::Create_LanczosResize, 0);
	env->AddFunction("Lanczos4ResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[threads]i",
		FilteredResizeMT::Create_Lanczos4Resize, 0);
	env->AddFunction("BlackmanResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[taps]i[threads]i",
		FilteredResizeMT::Create_BlackmanResize, 0);
	env->AddFunction("Spline16ResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[threads]i",
		FilteredResizeMT::Create_Spline16Resize, 0);
	env->AddFunction("Spline36ResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[threads]i",
		FilteredResizeMT::Create_Spline36Resize, 0);
	env->AddFunction("Spline64ResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[threads]i",
		FilteredResizeMT::Create_Spline64Resize, 0);
	env->AddFunction("GaussResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[p]f[threads]i",
		FilteredResizeMT::Create_GaussianResize, 0);
	env->AddFunction("SincResizeMT", "c[target_width]i[target_height]i[src_left]f[src_top]f[src_width]f[src_height]f[taps]i[threads]i",
		FilteredResizeMT::Create_SincResize, 0);

	return "ResizeMT plugin";
	
}


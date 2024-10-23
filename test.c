#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
// #include <time.h>
// #define TIME_UTC 1

#define M 131
const uint64_t f[3] = {0x0000000000002007, 0x0000000000000000, 0x0000000000000008};

#ifdef ONLINE_JUDGE
#define in stdin
#define out stdout
#endif

#ifdef __GNUC__
#pragma GCC target("pclmul,sse2")
#endif

// 以x^130 + x^129 + ...格式输出
void gf_print(const uint64_t a[5]);
// c = a + b
void gf_add(uint64_t res[5], const uint64_t a[5], const uint64_t b[5]);
// c = a * b
void gf_mul(uint64_t res[5], const uint64_t a[5], const uint64_t b[5]);
// c=a^n
void gf_pow(uint64_t res[5], const uint64_t a[5], uint64_t n);
// c = a^2
void gf_pow2(uint64_t res[5], const uint64_t a[5]);
// c = a^(-1)
void gf_inv(uint64_t res[5], const uint64_t a[5]);
// c = a mod f
void gf_mod(uint64_t res[5], const uint64_t a[5]);
// deg(a)
int degree(const uint64_t a[5]);
// res= a << d
void left_shift(uint64_t res[5], uint64_t const a[5], int d);

void gf_mul_trivial(uint64_t res[5], const uint64_t a[5], const uint64_t b[5]);
void gf_mul_bit(uint64_t res[5], const uint64_t a[5], const uint64_t b[5]);
void gf_mul_simd(uint64_t res[5], const uint64_t a[5], const uint64_t b[5]);
#ifndef gf_mul
// #define gf_mul (gf_mul_trivial)
// #define gf_mul (gf_mul_bit)
#define gf_mul (gf_mul_simd)
#endif

void gf_pow2_bit(uint64_t res[5], const uint64_t a[5]);
void gf_pow2_mul(uint64_t res[5], const uint64_t a[5]);
void gf_pow2_simd(uint64_t res[5], const uint64_t a[5]);
#ifndef gf_pow2
// #define gf_pow2 (gf_pow2_bit)
// #define gf_pow2 (gf_pow2_mul)
#define gf_pow2 (gf_pow2_simd)
#endif

void gf_pow_trivial(uint64_t res[5], const uint64_t a[5], uint64_t n);
void gf_pow_fast(uint64_t res[5], const uint64_t a[5], uint64_t n);
#ifndef gf_pow
// #define gf_pow (gf_pow_trivial)
#define gf_pow (gf_pow_fast)
#endif

void gf_inv_euclid(uint64_t res[5], const uint64_t a[5]);
void gf_inv_fermat(uint64_t res[5], const uint64_t a[5]);
#ifndef gf_inv
// #define gf_inv (gf_inv_euclid)
#define gf_inv (gf_inv_fermat)
#endif

void gf_print(const uint64_t a[5])
{
    int flag = 1;
    for (int i = 0; i < 5; i++)
    {
        uint64_t mask = 0x8000000000000000;
        for (int j = 0; j < 64; j++)
        {
            if (a[4 - i] & mask)
            {
                if (flag)
                {
                    printf("x^%d", (5 - i) * 64 - j - 1);
                    flag = 0;
                }
                else if ((5 - i) * 64 - j - 1 == 0)
                    printf("+1");
                else
                    printf("+x^%d", (5 - i) * 64 - j - 1);
            }
            mask >>= 1;
        }
    }
    printf("\n");
}

void gf_add(uint64_t res[5], const uint64_t a[5], const uint64_t b[5])
{
    for (int i = 0; i < 3; i++)
    {
        res[i] = a[i] ^ b[i];
    }
}

void gf_mul_trivial(uint64_t res[5], const uint64_t a[5], const uint64_t b[5])
{
    uint64_t tem[5] = {0};
    for (int i = 0; i < 131; i++)
    {
        for (int j = 0; j < 131; j++)
        {
            int i1 = i / 64, i2 = i % 64;
            int j1 = j / 64, j2 = j % 64;
            int ij1 = (i + j) / 64, ij2 = (i + j) % 64;

            const uint64_t mask = 1;
            uint64_t ai = 0, bj = 0;
            if (a[i1] & (mask << i2))
                ai = 1;
            if (b[j1] & (mask << j2))
                bj = 1;

            tem[ij1] ^= (ai & bj) << ij2;
        }
    }
    gf_mod(res, tem);
}

void gf_mul_simd(uint64_t res[5], const uint64_t a[5], const uint64_t b[5])
{
    __m128i A0 = _mm_set_epi64x(0, a[0]);
    __m128i A1 = _mm_set_epi64x(0, a[1]);
    __m128i A2 = _mm_set_epi64x(0, a[2]);
    __m128i B0 = _mm_set_epi64x(0, b[0]);
    __m128i B1 = _mm_set_epi64x(0, b[1]);
    __m128i B2 = _mm_set_epi64x(0, b[2]);

    __m128i S0 = _mm_xor_si128(A0, A1);
    __m128i S1 = _mm_xor_si128(A1, A2);
    __m128i S2 = _mm_xor_si128(A0, A2);
    __m128i S3 = _mm_xor_si128(B0, B1);
    __m128i S4 = _mm_xor_si128(B1, B2);
    __m128i S5 = _mm_xor_si128(B0, B2);

    __m128i P0 = _mm_clmulepi64_si128(A0, B0, 0x00);
    __m128i P1 = _mm_clmulepi64_si128(A1, B1, 0x00);
    __m128i P2 = _mm_clmulepi64_si128(A2, B2, 0x00);
    __m128i P3 = _mm_xor_si128(_mm_clmulepi64_si128(S0, S3, 0x00), _mm_xor_si128(P0, P1));
    __m128i P4 = _mm_xor_si128(_mm_clmulepi64_si128(S2, S5, 0x00), _mm_xor_si128(P0, P2));
    __m128i P5 = _mm_xor_si128(_mm_clmulepi64_si128(S1, S4, 0x00), _mm_xor_si128(P1, P2));

    uint64_t tem[5] = {0};
    tem[0] = _mm_extract_epi64(P0, 0);
    tem[1] = _mm_extract_epi64(P3, 0) ^ _mm_extract_epi64(P0, 1);
    tem[2] = _mm_extract_epi64(_mm_xor_si128(P1, P4), 0) ^ _mm_extract_epi64(P3, 1);
    tem[3] = _mm_extract_epi64(P5, 0) ^ _mm_extract_epi64(_mm_xor_si128(P1, P4), 1);
    tem[4] = _mm_extract_epi64(P2, 0) ^ _mm_extract_epi64(P5, 1);
    // tem[5] = _mm_extract_epi64(P2, 1);

    gf_mod(res, tem);
}

void gf_mul_bit(uint64_t res[5], const uint64_t a[5], const uint64_t b[5])
{
    uint64_t tem[5] = {0};
    uint64_t b_[5] = {0};
    for (int i = 0; i < 5; i++)
    {
        uint64_t mask = 1;
        for (int j = 0; j < 64; j++)
        {
            if (a[i] & mask)
            {

                left_shift(b_, b, i * 64 + j);
                for (int k = 0; k < 5; k++)
                    tem[k] = tem[k] ^ b_[k];
            }
            mask <<= 1;
        }
    }
    gf_mod(res, tem);
}

void gf_mod(uint64_t res[5], const uint64_t a[5])
{
    uint64_t e[5];
    for (int i = 0; i < 5; i++)
    {
        e[i] = a[i];
    }

    for (int i = 4; i >= 3; i--)
    {
        uint64_t t = e[i];
        e[i - 3] = e[i - 3] ^ (t << 61) ^ (t << 62) ^ (t << 63);
        e[i - 2] = e[i - 2] ^ (t << 10) ^ (t >> 1) ^ (t >> 2) ^ (t >> 3);
        e[i - 1] = e[i - 1] ^ (t >> 54);
    }

    uint64_t t = e[2] >> 3;
    res[0] = e[0] ^ (t << 13) ^ t ^ (t << 1) ^ (t << 2);
    res[1] = e[1] ^ (t >> 51);
    res[2] = e[2] & 0x7;
}

void gf_pow_trivial(uint64_t res[5], const uint64_t a[5], uint64_t n)
{
    uint64_t tem[5] = {1, 0, 0, 0, 0};
    for (uint64_t i = 0; i < n; i++)
    {
        gf_mul(tem, tem, a);
    }
    gf_mod(res, tem);
}

void gf_pow_fast(uint64_t res[5], const uint64_t a[5], uint64_t n)
{
    uint64_t tem[5] = {1, 0, 0, 0, 0};
    uint64_t base[5];
    for (int i = 0; i < 5; i++)
    {
        base[i] = a[i];
    }
    while (n)
    {
        if (n & 1)
        {
            gf_mul(tem, tem, base);
        }
        gf_pow2(base, base);
        n >>= 1;
    }
    gf_mod(res, tem);
}

void gf_pow2_mul(uint64_t res[5], const uint64_t a[5])
{
    gf_mul_simd(res, a, a);
}

void gf_pow2_bit(uint64_t res[5], const uint64_t a[5])
{
    uint64_t tem[5] = {0};
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            int i_ = ((i * 64 + j) * 2) / 64;
            int j_ = (i * 64 + j) * 2 % 64;

            uint64_t mask = 1;
            if (a[i] & (mask << j))
            {
                tem[i_] |= mask << j_;
            }
        }
    }
    gf_mod(res, tem);
}

void gf_pow2_simd(uint64_t res[5], const uint64_t a[5])
{
    __m128i A0 = _mm_set_epi64x(0, a[0]);
    __m128i A1 = _mm_set_epi64x(0, a[1]);
    __m128i A2 = _mm_set_epi64x(0, a[2]);

    __m128i P0 = _mm_clmulepi64_si128(A0, A0, 0x00);
    __m128i P1 = _mm_clmulepi64_si128(A1, A1, 0x00);
    __m128i P2 = _mm_clmulepi64_si128(A2, A2, 0x00);

    uint64_t tem[5] = {0};
    tem[0] = _mm_extract_epi64(P0, 0);
    tem[1] = _mm_extract_epi64(P0, 1);
    tem[2] = _mm_extract_epi64(P1, 0);
    tem[3] = _mm_extract_epi64(P1, 1);
    tem[4] = _mm_extract_epi64(P2, 0);
    // tem[5] = _mm_extract_epi64(P2, 1);

    gf_mod(res, tem);
}

void gf_inv_euclid(uint64_t res[5], const uint64_t a[5])
{
    uint64_t u[5] = {0}, v[5] = {0};
    for (int i = 0; i < 3; i++)
    {
        u[i] = a[i];
        v[i] = f[i];
    }

    uint64_t g1[5] = {1, 0, 0, 0, 0};
    uint64_t g2[5] = {0, 0, 0, 0, 0};
    uint64_t tem[5] = {0};
    while (degree(u))
    {
        int d = degree(u) - degree(v);

        if (d < 0)
        {
            for (int i = 0; i < 5; i++)
            {
                uint64_t tem = u[i];
                u[i] = v[i];
                v[i] = tem;
            }
            for (int i = 0; i < 5; i++)
            {
                uint64_t tem = g1[i];
                g1[i] = g2[i];
                g2[i] = tem;
            }
            d = -d;
        }

        left_shift(tem, v, d);
        for (int i = 0; i < 5; i++)
        {
            u[i] ^= tem[i];
        }

        left_shift(tem, g2, d);
        for (int i = 0; i < 5; i++)
        {
            g1[i] ^= tem[i];
        }
    }

    gf_mod(tem, g1);
    for (int i = 0; i < 5; i++)
    {
        res[i] = tem[i];
    }
}

void gf_inv_fermat(uint64_t res[5], const uint64_t a[5])
{
    uint64_t x_i[5];
    for (int i = 0; i < 5; i++)
        x_i[i] = a[i];

    uint64_t n = M - 1;
    int bits = 0;
    while (n)
    {
        n >>= 1;
        bits++;
    }

    n = M - 1;
    uint64_t mask = 1ULL << (bits - 2);
    int n_i = 1;
    for (int i = 0; i < bits - 1; i++)
    {
        uint64_t tem[5] = {0};
        for (int j = 0; j < 5; j++)
            tem[j] = x_i[j];
        if (n & mask)
        {
            for (int j = 0; j < n_i; j++)
                gf_pow2(tem, tem);
            gf_mul(tem, x_i, tem);
            gf_pow2(tem, tem);
            gf_mul(x_i, a, tem);
            n_i = n_i * 2 + 1;
        }
        else
        {
            for (int j = 0; j < n_i; j++)
                gf_pow2(tem, tem);
            gf_mul(x_i, x_i, tem);
            n_i *= 2;
        }
        mask >>= 1;
    }
    gf_pow2(res, x_i);
}

int degree(uint64_t const a[5])
{
    for (int i = 0; i < 5; i++)
    {
        uint64_t tem = 0x8000000000000000;
        for (int j = 0; j < 64; j++)
        {
            if (a[4 - i] & tem)
            {
                return (5 - i) * 64 - j - 1;
            }
            tem >>= 1;
        }
    }
    return -1;
}

void left_shift(uint64_t res[5], uint64_t const a[5], int d)
{
    if (d % 64 == 0)
    {
        int tem = d / 64;
        for (int i = 0; i < tem; i++)
        {
            res[i] = 0;
        }
        for (int i = tem; i < 5; i++)
        {
            res[i] = a[i - tem];
        }
    }
    else
    {
        if (d < 64)
        {
            for (int i = 4; i > 0; --i)
            {
                res[i] = (a[i] << d) | (a[i - 1] >> (64 - d));
            }
            res[0] = a[0] << d;
        }
        else if (d < 128)
        {
            d -= 64;
            for (int i = 4; i > 1; --i)
            {
                res[i] = (a[i - 1] << d) | (a[i - 2] >> (64 - d));
            }
            res[1] = a[0] << d;
            res[0] = 0;
        }
        else if (d < 192)
        {
            d -= 128;
            for (int i = 4; i > 2; --i)
            {
                res[i] = (a[i - 2] << d) | (a[i - 3] >> (64 - d));
            }
            res[2] = a[0] << d;
            res[1] = 0;
            res[0] = 0;
        }
    }
}

uint64_t now()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (uint64_t)1e9 * t.tv_sec + t.tv_nsec;
}

void test_mul_time(uint64_t res[5], uint64_t f1[5], uint64_t f2[5])
{
    for (int i = 0; i < 100; i++)
    {

        uint64_t ts = now();
        for (int j = 0; j < 10000; j++)
            gf_mul_trivial(res, f1, f2);
        uint64_t te = now();
        printf("mul_trivial: %lldns\n", (te - ts) / 10000);

        ts = now();
        for (int j = 0; j < 10000; j++)
            gf_mul_bit(res, f1, f2);
        te = now();
        printf("mul_bit: %lldns\n", (te - ts) / 10000);

        ts = now();
        for (int j = 0; j < 10000; j++)
            gf_mul_simd(res, f1, f2);
        te = now();
        printf("mul_smid: %lldns\n", (te - ts) / 10000);

        printf("\n");
    }
}

void test_pow2_time(uint64_t res[5], uint64_t f1[5])
{
    for (int i = 0; i < 100; i++)
    {
        uint64_t ts = now();
        for (int j = 0; j < 10000; j++)
            gf_pow2_bit(res, f1);
        uint64_t te = now();
        printf("pow2_bit: %lldns\n", (te - ts) / 10000);

        ts = now();
        for (int j = 0; j < 10000; j++)
            gf_pow2_mul(res, f1);
        te = now();
        printf("pow2_mul: %lldns\n", (te - ts) / 10000);

        printf("\n");

        ts = now();
        for (int j = 0; j < 10000; j++)
            gf_pow2_simd(res, f1);
        te = now();
        printf("pow2_simd: %lldns\n", (te - ts) / 10000);
    }
}

void test_inv_time(uint64_t res[5], uint64_t f1[5])
{
    for (int j = 0; j < 100; j++)
    {
        uint64_t ts = now();
        for (int i = 0; i < 10000; i++)
            gf_inv_euclid(res, f1);
        uint64_t te = now();
        printf("inv_euclid: %lldns\n", (te - ts) / 10000);

        ts = now();
        for (int i = 0; i < 10000; i++)
            gf_inv_fermat(res, f1);
        te = now();
        printf("inv_fermat: %lldns\n", (te - ts) / 10000);

        printf("\n");
    }
}

int main(void)
{
#ifndef ONLINE_JUDGE
    FILE *in = fopen("input2.bin", "rb");
    FILE *out = fopen("output.bin", "wb");
#endif

    uint32_t count;
    fread(&count, sizeof(count), 1, in);
    while (count--)
    {
        uint8_t type;
        fread(&type, sizeof(type), 1, in);

        uint64_t f1[5] = {0};
        for (int j = 0; j < 3; j++)
            fread(&f1[j], sizeof(f1[j]), 1, in);
        uint64_t f2[5] = {0};
        for (int j = 0; j < 3; j++)
            fread(&f2[j], sizeof(f2[j]), 1, in);

        uint64_t res[5] = {0};
        // test_mul_time(res, f1, f2);
        // test_pow2_time(res, f1);
        // test_inv_time(res, f1);

        for (int i = 0; i < 5; i++)
            res[i] = 0;

        if (type == 0x00)
        {
            gf_add(res, f1, f2);
            for (int i = 0; i < 3; i++)
                fwrite(&res[i], sizeof(res[i]), 1, out);
        }
        else if (type == 0x01)
        {
            gf_mul(res, f1, f2);
            for (int i = 0; i < 3; i++)
                fwrite(&res[i], sizeof(res[i]), 1, out);
        }
        else if (type == 0x02)
        {
            gf_pow2(res, f1);
            for (int i = 0; i < 3; i++)
                fwrite(&res[i], sizeof(res[i]), 1, out);
        }
        else if (type == 0x03)
        {
            gf_inv(res, f1);
            for (int i = 0; i < 3; i++)
                fwrite(&res[i], sizeof(res[i]), 1, out);
        }
    }
}
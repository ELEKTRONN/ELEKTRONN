/*
ELEKTRONN - Neural Network Toolkit
Copyright (c) 2015 Gregor Urban, Marius Killinger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software for non-commercial purposes, including
the rights to use, copy, modify, merge, publish, distribute, the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The Software shall neither be used as part or facilitating factor of military
applications, nor be used to develop or facilitate the development
of military applications.
The Software and derivative work is not used commercially.
The above copyright notice and this permission notice shall be included in
all copies and all derivative work of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

typedef int bool;
#define True 1
#define False 0
typedef float (*interpolateFunc2d)(float *, float, float, int, int, int);

// Linear Interpolation
/// Returns either interpolation of both values if available, or single if available or 0 if not avail
inline float linearInterpolate(float p[2], bool avail[2], float x) {
    float ret = 0;
    if (avail[0]) {
        if (avail[1])
            ret = (1 - x) * p[0] + x * p[1];
        else
            ret = p[0];
    } else if (avail[1])
        ret = p[1];
    return ret;
}

inline float bilinearInterpolate(const float *src, float u, float v, int w_x, int w_y) {
    int x_l = floor(u);
    int x_r = ceil(u);
    int y_l = floor(v);
    int y_r = ceil(v);

    float ret;

    float p[2][2] = {{0, 0}, {0, 0}}; // ll, lr, rl, rr
    bool avail[2][2] = {{False, False}, {False, False}};

    if (x_l < w_x && y_l < w_y && x_l >= 0 && y_l >= 0) {
        p[0][0] = src[x_l * w_y + y_l];
        avail[0][0] = True;
    }
    if (x_l < w_x && y_r < w_y && x_l >= 0 && y_r >= 0) {
        p[0][1] = src[x_l * w_y + y_r];
        avail[0][1] = True;
    }
    if (x_r < w_x && y_l < w_y && x_r >= 0 && y_l >= 0) {
        p[1][0] = src[x_r * w_y + y_l];
        avail[1][0] = True;
    }
    if (x_r < w_x && y_r < w_y && x_r >= 0 && y_r >= 0) {
        p[1][1] = src[x_r * w_y + y_r];
        avail[1][1] = True;
    }

    if (!(avail[0][0] || avail[0][1] || avail[1][0] || avail[1][1]))
        ret = 0;
    else {
        float x = u - x_l;
        float y = v - y_l;
        // ret=bilinearInterpolate(p, avail, dx, dy);
        float tmp[2];
        tmp[0] = linearInterpolate(p[0], avail[0], y);
        tmp[1] = linearInterpolate(p[1], avail[1], y);

        bool avail_tmp[2] = {False, False};
        if (avail[0][0] || avail[0][1])
            avail_tmp[0] = True;
        if (avail[1][0] || avail[1][1])
            avail_tmp[1] = True;
        ret = linearInterpolate(tmp, avail_tmp, x);
    }
    return ret;
}

/************************************************************************************************************/

void NN2d(const float *src, float u, float v, int ch, const int sh[3],
                  const int strd_src[2], float *ret) {
    int x = trunc(u + 0.5);
    int y = trunc(v + 0.5);
    if (x >= sh[1] || y >= sh[2] || x < 0 || y < 0) {
        *ret = 0;
    } else {
        *ret = src[ch * strd_src[0] + x * strd_src[1] + y];
    }
}

int fastwarp2d_opt(const float *src, float *dest_d, const int sh[3],
                   const int ps[3], const float rot, const float shear,
                   const float scale[2], const float stretch_in[2]) {
    // Loop/coord indices
    int i, j, ch; // pixel index in dest
    float xt, yt; // Intermediate coordinates
    float u, v;   // pixel coordinate in src (mapping origin)

    float x_center_off = (float)sh[1] / 2 - 0.5; // used to center coordinates
    float y_center_off = (float)sh[2] / 2 - 0.5;
    float x, y; // center pixel index in dest (because it is centered it may  x.5!)
    // the source coordinates u,v calculated from x,y must be 'de-centered'

    int strd[2] = {ps[1] * ps[2], ps[2]};
    int strd_src[2] = {sh[1] * sh[2], sh[2]};
    // Parameter constant handling
    float stretch[2];
    stretch[0] = stretch_in[0] / x_center_off;
    stretch[1] = stretch_in[1] / y_center_off;

    // Loop Optimisation
    float ret;
    float sin_plus = sin(rot + shear);
    float cos_plus = cos(rot + shear);
    float sin_minu = sin(rot - shear);
    float cos_minu = cos(rot - shear);
    // printf("dx=%.4f dy=%.4f\n", x_center_off, y_center_off);
    // printf("(%i, %i %i)", sh[0], sh[1], sh[2]);
    for (ch = 0; ch < sh[0]; ch++) {
        x = -x_center_off + (sh[1] - ps[1]) / 2;
        for (i = 0; i < ps[1]; i++) {
            y = -y_center_off + (sh[2] - ps[2]) / 2;
            for (j = 0; j < ps[2]; j++) {
                xt = x * (scale[0] + stretch[0] * y); // /sqrt(1+abs(x)));
                yt = y * (scale[1] + stretch[1] * x); // /sqrt(1+abs(y)));
                u = xt * cos_minu - yt * sin_plus + x_center_off;
                v = yt * cos_plus + xt * sin_minu + y_center_off;

                NN2d(src, u, v, ch, sh, strd_src, &ret);
                dest_d[ch * strd[0] + i * strd[1] + j] = ret;

                // printf("WRITE dest_d[%9i]\n", ch*ps[0]*ps[1]*ps[2] + i*ps[1]*ps[2] + j*ps[2] + k);
                // printf("u=%2.1f v=%2.1f x=%2.1f y=%2.1f i=%02i j=%02i val=%.1f\n", u, v, x, y, i, j, ret);
                y++;
            }
            x++;
        }
    }
    return 0;
}

/************************************************************************************************************/
void NN3d_zxy(const float *src, float u, float v, float w, int ch,
                      const int sh[4], const int strd_src[3], float *ret) {
    int x = trunc(u + 0.5);
    int y = trunc(v + 0.5);
    int z = trunc(w + 0.5);
    if (x >= sh[2] || y >= sh[3] || x < 0 || y < 0 || z >= sh[0] || z < 0) {
        *ret = 0;
    } else {
        *ret = src[z * strd_src[0] + ch * strd_src[1] + x * strd_src[2] + y];
    }
}

int fastwarp3d_opt_zxy(const float *src, float *dest_d,
                       const int sh[4], // z,ch,x,y
                       const int ps[4], // z,ch, x,y
                       const float rot, const float shear, const float scale[3],
                       const float stretch_in[4], const float twist_in) {
    // Loop/coord indices
    int i, j, k, ch; // pixel index in dest
    float xt, yt;    // Intermediate coordinates
    float u, v, w;   // pixel coordinate in src (mapping origin)

    float x_center_off = (float)sh[2] / 2 - 0.5; // used to center coordinates
    float y_center_off = (float)sh[3] / 2 - 0.5;
    float z_center_off = (float)sh[0] / 2 - 0.5;
    float x, y, z; // center pixel index in dest (because it is centered it may  x.5!)
    // the source coordinates u,v calculated from x,y must be 'de-centered'

    int strd[3] = {ps[1] * ps[2] * ps[3], ps[2] * ps[3], ps[3]};
    int strd_src[3] = {sh[1] * sh[2] * sh[3], sh[2] * sh[3], sh[3]};
    // Parameter constant handling
    float stretch[4];
    stretch[0] = stretch_in[0] / x_center_off;
    stretch[1] = stretch_in[1] / y_center_off;
    stretch[2] = stretch_in[2] / z_center_off;
    stretch[3] = stretch_in[3] / z_center_off;
    float twist = twist_in / z_center_off;

    // Loop Optimisation
    float ret;
    float sin_plus;
    float cos_plus;
    float sin_minu;
    float cos_minu;

    z = -z_center_off + (sh[0] - ps[0]) / 2;
    for (k = 0; k < ps[0]; k++) {
        sin_plus = sin(rot + shear + z * twist);
        cos_plus = cos(rot + shear + z * twist);
        sin_minu = sin(rot - shear + z * twist);
        cos_minu = cos(rot - shear + z * twist);
        w = z * scale[2] + z_center_off;
        for (ch = 0; ch < sh[1]; ch++) {
            x = -x_center_off + (sh[2] - ps[2]) / 2;
            for (i = 0; i < ps[2]; i++) {
                y = -y_center_off + (sh[3] - ps[3]) / 2;
                for (j = 0; j < ps[3]; j++) {
                    xt = x * (scale[0] + stretch[0] * y + stretch[2] * z);
                    yt = y * (scale[1] + stretch[1] * x + stretch[3] * z);
                    u = xt * cos_minu - yt * sin_plus + x_center_off;
                    v = yt * cos_plus + xt * sin_minu + y_center_off;

                    NN3d_zxy(src, u, v, w, ch, sh, strd_src, &ret);
                    dest_d[k * strd[0] + ch * strd[1] + i * strd[2] + j] = ret;

                    // printf("WRITE dest_d[%9i]\n", ch*ps[0]*ps[1]*ps[2] + i*ps[1]*ps[2] + j*ps[2] + k);
                    // printf("u=%.1f v=%.1f w=%.1f x=%.1f y=%.1f z=%.1f i=%i j=%i k=%i val=%.1f\n", u, v, w, x, y, z, i, j, k, ret);
                    y++;
                }
                x++;
            }
        }
        z++; // put into loop from above!?
    }
    return 0;
}

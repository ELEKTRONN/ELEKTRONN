/*
ELEKTRONN - Neural Network Toolkit
Copyright (c) 2014 - now
Max-Planck-Institute for Medical Research, Heidelberg, Germany
Authors: Marius Killinger, Gregor Urban
*/

#include <math.h>


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


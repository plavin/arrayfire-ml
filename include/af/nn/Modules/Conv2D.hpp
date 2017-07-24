/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/nn/Modules/Module.hpp>

namespace af
{
    namespace nn
    {
        class Convolution2D : public Module
        {
        private:
            bool m_bias;
            int m_px;
            int m_py;
            int m_sx;
            int m_sy;
            int m_px;
            int m_py;
        public:
            Convolution2D(int wy, int wy, int sx, int sy, int px, int py, int n_in, int n_out, bool bias = true, float spread = 0.05);

            Convolution2D(const autograd::Variable &w);

            Convolution2D(const autograd::Variable &w, const autograd::Variable &b);

            autograd::Variable forward(const autograd::Variable &input);
        };
    }
}

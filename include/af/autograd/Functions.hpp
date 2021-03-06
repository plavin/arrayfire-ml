/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

namespace af {
    namespace autograd {

        class Variable;

        Variable operator +(const Variable &lhs, const Variable &rhs);
        Variable operator *(const Variable &lhs, const Variable &rhs);
        Variable operator -(const Variable &lhs, const Variable &rhs);
        Variable operator /(const Variable &lhs, const Variable &rhs);

        Variable operator +(const double &lhs, const Variable &rhs);
        Variable operator *(const double &lhs, const Variable &rhs);
        Variable operator -(const double &lhs, const Variable &rhs);
        Variable operator /(const double &lhs, const Variable &rhs);

        Variable operator +(const Variable &lhs, const double &rhs);
        Variable operator *(const Variable &lhs, const double &rhs);
        Variable operator -(const Variable &lhs, const double &rhs);
        Variable operator /(const Variable &lhs, const double &rhs);

        Variable negate(const Variable &input);
        Variable reciprocal(const Variable &input);

        Variable exp(const Variable &input);
        Variable sin(const Variable &input);
        Variable cos(const Variable &input);
        Variable tanh(const Variable &input);
        Variable sigmoid(const Variable &input);
      
        Variable max(const Variable &lhs, const Variable &rhs);
        Variable max(const Variable &lhs, const double &rhs);
        Variable max(const double &lhs, const Variable &rhs);
      
        Variable min(const Variable &lhs, const Variable &rhs);
        Variable min(const Variable &lhs, const double &rhs);
        Variable min(const double &lhs, const Variable &rhs);
      
        Variable transpose(const Variable &input);
        Variable expandAs(const Variable &input, const Variable &reference);
        Variable reduceAs(const Variable &input, const Variable &reference);

        Variable matmul(const Variable &lhs, const Variable &rhs);
        Variable matmulTN(const Variable &lhs, const Variable &rhs);
        Variable matmulNT(const Variable &lhs, const Variable &rhs);


    }
}

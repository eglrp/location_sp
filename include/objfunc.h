/*
 * Copyright (c) 2008-2011 Zhang Ming (M. Zhang), zmjerry@163.com
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 2 or any later version.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details. A copy of the GNU General Public License is available at:
 * http://www.fsf.org/licensing/licenses
 */


/*****************************************************************************
 *                                objfunc.h
 *
 * Function Object.
 *
 * We can use this functor computing the value of objective function and it's
 * gradient vector. The objective function is supposed to be multidimensional,
 * one dimention is the special case of "vector.dim()=1".
 *
 * Zhang Ming, 2010-03, Xi'an Jiaotong University.
 *****************************************************************************/


#ifndef OBJFUNC_H
#define OBJFUNC_H


#include <vector.h>


namespace splab
{

    template <typename Type>
    class ObjFunc
    {

    public:

        /**
         * Initialize the parameters
         */
        ObjFunc( Vector<Type> aa, Vector<Type> bb, Vector<Type> cc, Vector<Type> dis) : a(aa), b(bb), c(cc), d(dis)
        { }

        /**
         * Compute the value of objective function at point x.
         */
        Type operator()( Vector<Type> &x )
        {
            Type tmp = Type(0);
            for(int i = 0; i < a.dim(); i++)
            {
                tmp += fabs((a[i] - x(1))*(a[i] - x(1)) + (b[i] - x(2))*(b[i] - x(2)) + (c[i] - x(3))*(c[i] - x(3)) - d[i]*d[i]);
            }

            return tmp / a.dim();
        }

        /**
         * Compute the gradient of objective function at point x.
         */
        Vector<Type> grad( Vector<Type> &x )
        {
            Vector<Type> df(x.dim());
            df(1) = Type(0);
            df(2) = Type(0);
            df(3) = Type(0);

            for(int i = 0; i < a.dim(); i++)
            {
                Type tmp = (a[i] - x(1))*(a[i] - x(1)) + (b[i] - x(2))*(b[i] - x(2)) + (c[i] - x(3))*(c[i] - x(3)) - d[i]*d[i];
                Type sign;
                if(tmp < 0) sign = -1;
                else if(tmp == 0) sign = 0;
                else sign = 1;

                df(1) += 2*sign*(x(1) - a[i]);
                df(2) += 2*sign*(x(2) - b[i]);
                df(3) += 2*sign*(x(3) - c[i]);
            }

            df /= Type(a.dim());

            return df;
        }

    private:

        // parameters
        Vector<Type> a,
                     b,
                     c,
                     d;

    };
    // class ObjFunc

}
// namespace splab


#endif
// OBJFUNC_H

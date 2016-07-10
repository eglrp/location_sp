#include <QCoreApplication>

#define BOUNDS_CHECK

#include <iostream>
#include <iomanip>
#include <objfunc.h>
#include <steepdesc.h>
#include <bfgs.h>
#include <timing.h>

using namespace std;
using namespace splab;

typedef double  Type;

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    Type aa[] = {0,4000,4000,0};
    Type bb[] = {0,0,4000,4000};
    Type cc[] = {3000,3000,3000,3000};
    Type dis[] = {2.940180186013095e+03,
                  4.864222396871262e+03,
                  6.017030789868371e+03,
                  4.603114111796926e+03};

    Vector<Type> a(4,aa), b(4,bb), c(4,cc), d(4,dis);

    ObjFunc<Type> f( a, b, c, d );
    Vector<Type> x0(3);

//    x0(1) = 2000;
//    x0(2) = 2000;
//    x0(3) = 2900;
    cout << x0 << endl;

    Type tolErr = 1e-3;
//    SteepDesc< Type, ObjFunc<Type> > steep;         //最速下降法
    BFGS< Type, ObjFunc<Type> > steep;              //高斯牛顿法

    Timing time;
    time.start();
    steep.optimize( f, x0, tolErr);                 //迭代计算
    time.stop();
    cout << "The running time is : " << time.read() << endl << endl;

    if( steep.isSuccess() )
    {
        Vector<Type> xmin = steep.getOptValue();
        int N = steep.getItrNum();
        cout << "The iterative number is:   " << N << endl << endl;
        cout << "The number of function calculation is:   "
             << steep.getFuncNum() << endl << endl;
        cout << setiosflags(ios::fixed) << setprecision(4);
        cout << "The optimal value of x is:   " << xmin << endl;
        cout << "The minimum value of f(x) is:   " << f(xmin) << endl << endl;
        cout << "The gradient's norm at x is:   "
             << steep.getGradNorm()[N] << endl << endl;
    }
    else
        cout << "The optimal solution  can't be found!" << endl;

    return app.exec();
}

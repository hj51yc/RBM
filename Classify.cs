using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;

namespace RBM
{
    class Classify
    {
        public static bool IsClassifyPassForMnist(double[]expect,double[] actual )
        {
            if(expect.Length!=actual.Length || expect.Length<1)
            {
                return false;
            }
            int expect_index = 0;
            int actual_index = 0;
            double max = expect[0];
            for (int i = 1; i < expect.Length; ++i)
            {
                if (max < expect[i])
                {
                    expect_index = i;
                    max = expect[i];
                }
            }
            max = actual[0];
            for (int i = 1; i < actual.Length; ++i)
            {
                if (max < actual[i])
                {
                    actual_index = i;
                    max = actual[i];
                }
            }
            if (expect_index == actual_index)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}

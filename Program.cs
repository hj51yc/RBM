using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
namespace RBM
{
    class Program
    {
        static void Main(string[] args)
        {
             DataProcess dp = new DataProcess();
            String trainFile = "D:/data/mnist/train-images.idx3-ubyte";
            String trainLabelFile = "D:/data/mnist/train-labels.idx1-ubyte";
            String testFile = "D:/data/mnist/t10k-images.idx3-ubyte";
            String testLabelFile = "D:/data/mnist/t10k-labels.idx1-ubyte";
            String saveWeightFile = "D:/data/mnist/RBM.weights";
            Console.WriteLine("Start to extract data.....");
            dp.ExtractMnistTrainsAndTests(trainFile, trainLabelFile, testFile, testLabelFile);
            dp.NormalizeTheData();
            dp.Shuffle();
            Console.WriteLine("extract data finished!!!");

            int TRAIN_SAMPLE_NUM=dp.trainDatas.Length;
            int TEST_SAMPLE_NUM = dp.testDatas.Length;
            int SAMPLE_DIMENSION = dp.trainDatas[0].Length;
            int HIDEN_NUM = 50;
            int CD_K=1;

         
            double[][] insamples =dp.trainDatas;
            double[][] outsamples = dp.trainLabels;

            double[][] testInSamples = dp.testDatas;
            double[][] testOutSamples = dp.testLabels;
         

            bool  convergenceFlag=false; //表示结束训练时状态，true表示训练满足阈值要求而结束，false表示round次数达到限制
            int MAX_ROUND=4; //最大的训练次数
            
            double CONVERGENCE_GRADIENT_NORM = 0.0000000001;//当梯度的范数小于这个大小的时候，说明梯度没有明显变化，则停止迭代。
            int THREAD_NUM = 4;
            double curGradientNorm = 0;

            int round=0;

            Stopwatch sw = new Stopwatch();
            sw.Start();
            Stopwatch sw_round = new Stopwatch();
           
            RBM_CD rbm=new RBM_CD(SAMPLE_DIMENSION,HIDEN_NUM,false,CD_K);
            rbm.InitParameterRandom();
            Console.WriteLine("begin find eta:");
            double t0 = rbm.DeterminInitT0(insamples,0,1000);
            rbm.SetT0(t0);
            Console.WriteLine("training start:");
            double[] costs = new double[MAX_ROUND];
            rbm.Train(insamples,testInSamples,CD_K,MAX_ROUND);
            sw.Stop();
            rbm.SaveWeightsToFile(saveWeightFile);
            System.Console.WriteLine("Training over!");
            if (!convergenceFlag)
            {
                System.Console.WriteLine("round time overflow , still not visit threshold!");
            }
            else
            {
                System.Console.WriteLine("convergence in "+round+" round ");
            }
            Console.WriteLine("总时间："+sw.Elapsed);
            Console.WriteLine("测量实例得出的总运行时间（毫秒为单位）：" + sw.ElapsedMilliseconds);
            Console.WriteLine("总运行时间(计时器刻度标识)：" + sw.ElapsedTicks);
            Console.WriteLine("计时器是否运行：" + sw.IsRunning.ToString());

            /*
            double []results=null;
            bool clsRight = false;
            int totalTestNum = dp.testDatas.Length;
            int clsRightCount = 0;
            for (int i = 0; i < totalTestNum; ++i)
            {
                results = neural.ComputeResultThroughNN(dp.testDatas[i]);
                clsRight = Classify.IsClassifyPassForMnist(dp.testLabels[i], results);
                if (clsRight)
                {
                    ++clsRightCount;
                }
            }
            Console.WriteLine("classfy correct rate:"+(((double)clsRightCount)/totalTestNum));
            Console.Read();
            */
        }
    }
}

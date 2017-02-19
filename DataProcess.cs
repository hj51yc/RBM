using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RBM
{
    class DataProcess
    {
        public double [][] trainDatas = null;
        public double[][] trainLabels = null;
        public double[][] testDatas = null;
        public double[][] testLabels = null;
        private int ReverseToBigEndian(int v,bool isBigEndian)
        {
            if (isBigEndian)
            {
                return v;
            }
            else
            {
                return BitConverter.ToInt32(BitConverter.GetBytes(v).Reverse().ToArray(),0);
            }
        }
        public bool ExtractMnist(String digitsFile, String labelsFile, out double[][] trains, out double[][] labels)
        {
            trains = null;
            labels = null;
            FileStream fs_digists = new FileStream(digitsFile, FileMode.Open, FileAccess.Read);
            BinaryReader breadDigits = new BinaryReader(fs_digists);
            FileStream fs_labels = new FileStream(labelsFile, FileMode.Open, FileAccess.Read);
            BinaryReader breadLabels = new BinaryReader(fs_labels);
            bool flag = true;
            int labelMagicNum = 2049;
            int digitsMagicNum = 2051;
            int lmn = breadLabels.ReadInt32();
            int dmn = breadDigits.ReadInt32();

            bool isBigEndian = !BitConverter.IsLittleEndian;
            if (ReverseToBigEndian(dmn, isBigEndian) != digitsMagicNum || ReverseToBigEndian(lmn, isBigEndian) != labelMagicNum)
            {
                flag = false;
            }
            else
            {
                int sampleNum =ReverseToBigEndian(breadDigits.ReadInt32(),isBigEndian);
                int rows = ReverseToBigEndian(breadDigits.ReadInt32(),isBigEndian);
                int cols = ReverseToBigEndian(breadDigits.ReadInt32(),isBigEndian);
                int dimens=rows*cols;
                int readNum=0;
                if (ReverseToBigEndian(breadLabels.ReadInt32(),isBigEndian) != sampleNum)
                {
                    flag = false;
                }
                else
                {
                    trains=new double[sampleNum][];
                    labels = new double[sampleNum][];
                    while (readNum < sampleNum)
                    {
                        trains[readNum] = new double[dimens];
                        labels[readNum] =new double[10];
                        int index=breadLabels.ReadByte();
                        for (int i = 0; i < 10; ++i)
                        {
                            labels[readNum][i] = 0.0;
                        }
                        labels[readNum][index] = 1.0;
                        for (int i = 0; i < dimens; ++i)
                        {
                            trains[readNum][i] = breadDigits.ReadByte();
                        }
                        ++readNum;
                    }
                }
            }

            breadLabels.Close();
            fs_digists.Close();
            breadDigits.Close();
            fs_labels.Close();
            return flag;
        }

        public void ExtractMnistTrainsAndTests(String trainDigitsFile, String trainLabelsFile,String testDigitsFile,String testLabelsFile)
        {

            if (!ExtractMnist(trainDigitsFile, trainLabelsFile, out this.trainDatas, out this.trainLabels))
            {
                System.Console.WriteLine("the training file invalid!");
            }
            if (!ExtractMnist(testDigitsFile, testLabelsFile, out this.testDatas, out testLabels))
            {
                System.Console.WriteLine("the test file invalid!");
            }
        }
        /*
        public void NormalizeTheData()
        {
            for (int i = 0; i < trainDatas.Length; ++i)
            {
                double len = 0.0;
                for (int j = 0; j < trainDatas[i].Length; ++j)
                {
                    len += trainDatas[i][j] * trainDatas[i][j];
                }
                len = Math.Sqrt(len);
                for (int j = 0; j < trainDatas[i].Length; ++j)
                {
                    trainDatas[i][j] /= len;
                }
            }
            for (int i = 0; i < testDatas.Length; ++i)
            {
                double len = 0.0;
                for (int j = 0; j < testDatas[i].Length; ++j)
                {
                    len += testDatas[i][j] * testDatas[i][j];
                }
                len = Math.Sqrt(len);
                for (int j = 0; j < testDatas[i].Length; ++j)
                {
                    testDatas[i][j] /= len;
                }
            }
        }
         */

        public void NormalizeTheData()
        {
            for (int i = 0; i < trainDatas.Length; ++i)
            {
                for (int j = 0; j < trainDatas[i].Length; ++j)
                {
                    trainDatas[i][j] /= 255;
                }
            }
            for (int i = 0; i < testDatas.Length; ++i)
            {
                for (int j = 0; j < testDatas[i].Length; ++j)
                {
                    testDatas[i][j] /= 255;
                }
            }
        }
        public void Shuffle()
        {
            Random random = new Random();
            int len = trainDatas.GetLength(0);
            for (int i = 0; i < len; i++)
            {
                int idx = random.Next(i, len);
                double[] tmp = trainDatas[i];
                trainDatas[i] = trainDatas[idx];
                trainDatas[idx] = tmp;
                tmp = trainLabels[i];
                trainLabels[i] = trainLabels[idx];
                trainLabels[idx] = tmp;
            }
            len = testDatas.GetLength(0);
            for (int i = 0; i < len; i++)
            {
                int idx = random.Next(i, len);
                double[] tmp = testDatas[i];
                testDatas[i] = testDatas[idx];
                testDatas[idx] = tmp;
                tmp = testLabels[i];
                testLabels[i] = testLabels[idx];
                testLabels[idx] = tmp;
            }
        }
        
    }
}

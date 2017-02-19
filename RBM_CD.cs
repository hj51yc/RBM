using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RBM
{
    class RBM_CD
    {
        /*
         *  eneger function: biasV'*V+biasH'*H+V*W*H;
         */
        double[,] m_W = null;
        double[] m_biasV = null;
        double[] m_biasH = null;

        double[,] m_WGradient = null;
        double[] m_biasVGradient = null;
        double[] m_biasHGradient = null;

        int m_VisibleDim;
        int m_HiddenDim;

        double[] m_PCD = null;
        bool m_persistent;//是否支持pcd,默认由h开始构造，m_PCD中保留构造过程中的h
        int m_CD_K;

        private double m_t0=200;
        private double m_eta0=1000;

        public RBM_CD(int vdim,int hdim,bool p,int CD_K)
        {
            this.m_VisibleDim=vdim;
            this.m_HiddenDim=hdim;
            this.m_W=new double[vdim,hdim];
            m_WGradient = new double[vdim, hdim];
            this.m_biasV=new double[vdim];
            this.m_biasH=new double[hdim];
            this.m_biasVGradient = new double[vdim];
            this.m_biasHGradient = new double[hdim];
            m_persistent = p;
            m_CD_K = CD_K;
            m_eta0 = 1000;
           
        }
        public double[] CalHiddenProbByVisible(double[] visible)
        {
            double[] probs = new double[m_HiddenDim];
            for (int i = 0; i < m_HiddenDim; ++i)
            {
                double tmp = 0.0;
                for (int j = 0; j < m_VisibleDim; ++j)
                {
                    tmp+=visible[j]*m_W[j,i];
                }
                tmp += m_biasH[i];
                probs[i] = tmp;
            }
            probs = Sigmoid(probs);
            return probs;
        }

        public double[] CalVisibleProbByHidden(double []hidden)
        {
            double []probs=new double[m_VisibleDim];
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                double tmp = 0.0;
                for (int j = 0; j < m_HiddenDim; ++j)
                {
                    tmp += m_W[i, j] * hidden[j];
                }
                tmp += m_biasV[i];
                probs[i] = tmp;
            }
            probs=Sigmoid(probs);
            return probs;
        }

        public double[] Sigmoid(double [] vec)
        {
            int len = vec.Length;
            double []result=new double[len];
            for (int i = 0; i < len; ++i)
            {
                result[i] = 1.0 / (1 + Math.Exp(-vec[i]));
            }
            return result;
        }

        double[] BinomialSampleVector(double[]successProb)
        {
            double[] results = new double[successProb.Length];
            Random r = new Random();
            for (int i = 0; i < successProb.Length; ++i)
            {
                double rand = r.NextDouble();
                if (rand > successProb[i])
                {
                   results[i]= 0;
                }
                else
                {
                   results[i]= 1.0;
                }
            }
            return results;
        }
        double[] GibbsVGivenH(double[] hsample)
        {
            double []V = CalVisibleProbByHidden(hsample);
            V = BinomialSampleVector(V);
            return V;
        }
        double[] GibbsHGivenV(double[] vsample)
        {
            double[] H = CalHiddenProbByVisible(vsample);
            H = BinomialSampleVector(H);
            return H;
        }
        public void InitParameterRandom()
        {
            /*
             * 随机数在[-2，2] 之间
             */
            Random r=new Random();
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                for (int j = 0; j < m_HiddenDim; ++j)
                {
                    m_W[i, j] =(r.NextDouble()-0.5)*4;
                }
            }
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                m_biasV[i] = (r.NextDouble() - 0.5) * 4; 
            }
            for (int j = 0; j < m_HiddenDim; ++j)
            {
                m_biasH[j] = (r.NextDouble() - 0.5) * 4; 
            }
        }

        public void TrainByOneSample(double[] vsample, int cdk, double learnRate)//CDK，其中K等于1,learning rate should use SGD 
        {
            double[] chainStart = null;

            double[] startV = vsample;
            double[] endV = null;

            double[] prob_h0_given_v0 = null;
            double[] prob_h1_given_v1 = null;
            prob_h0_given_v0 = CalHiddenProbByVisible(vsample);
            if (m_persistent)
            {
                chainStart = m_PCD;
                
            }
            else
            {
                chainStart = BinomialSampleVector(prob_h0_given_v0);
            }
            for (int i = 0; i < cdk; ++i)
            {
                endV = GibbsVGivenH(chainStart);
                prob_h1_given_v1 = CalHiddenProbByVisible(endV);
                chainStart = BinomialSampleVector(prob_h1_given_v1);
            }

            //calculate w gradient
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                for (int j = 0; j < m_HiddenDim; ++j)
                {
                    m_WGradient[i, j] = prob_h0_given_v0[j] * startV[i] - endV[i] * prob_h1_given_v1[j];
                    m_W[i,j]+=learnRate*m_WGradient[i,j];
                }
            }
            //calculate biasV gradient
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                m_biasVGradient[i] = startV[i] - endV[i];
                m_biasV[i]+=learnRate*m_biasVGradient[i];
            }
            //calculate biasH gradient
            for (int i = 0; i < m_HiddenDim; ++i)
            {
                m_biasHGradient[i]=prob_h0_given_v0[i]-prob_h1_given_v1[i];
                m_biasH[i]+=learnRate*m_biasHGradient[i];
            }

        }

        public void Train(double[][] trainSamples,double [][] testSamples,int CD_K,int ITERATE_NUM)
        {
            int iterate = 0;
            double r = m_eta0 / m_t0;
            int t = 0;

            while (iterate < ITERATE_NUM)
            {
                ++iterate;
                System.Console.WriteLine("iterate:" + iterate);
                for (int i = 0; i < trainSamples.Length; ++i)
                {
                    r = m_eta0 / (m_t0+t);
                    TrainByOneSample(trainSamples[i], CD_K, r);
                    ++t;
                    //System.Console.WriteLine("t:" + t);
                }
                double err = TestAllError(testSamples);
                System.Console.WriteLine("t:" + t + "\t error:" + err);
            }
        }
        public void SetT0(double t0)
        {
            this.m_t0 = t0;
        }
        double FreeEnerge(double[] v)
        {
            double vbiasTerm = 0.0;
            double[] wh_hbias = new double[m_HiddenDim];
            double hiddenTerm = 0.0;
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                vbiasTerm += m_biasV[i] * v[i];
            }
            for (int i = 0; i < m_HiddenDim; ++i)
            {
                wh_hbias[i] = m_biasH[i];
                for (int j = 0; j < m_VisibleDim; ++j)
                {
                    wh_hbias[i] += v[j] * m_W[j, i];
                }
            }

            for (int i = 0; i < m_HiddenDim; ++i)
            {
                hiddenTerm += Math.Log(1 + Math.Exp(wh_hbias[i]));
            }
            return -vbiasTerm - hiddenTerm;
        }
        public double TestReconstructionError(double[] v)
        {
            double[] H = CalHiddenProbByVisible(v);
            H = BinomialSampleVector(H);//this one can be taken away
            double[] V_reconstruct = CalVisibleProbByHidden(H);
            double error = 0.0;
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                error += Math.Pow(V_reconstruct[i] - v[i], 2);
            }
            return error;
        }
        public double TestAllError(double[][] samples)
        {
            double error = 0;
            for (int i = 0; i < samples.Length; ++i)
            {
                error +=TestReconstructionError(samples[i]);
            }
            return error;
        }

        public double EvaluteCostByT0(double[][] X, int start, int end, double t0)
        {
            //Console.WriteLine("start evaluate eta:" + t0);
            double t = 0;
            for (int i = start; i < end; ++i)
            {
                ++t;
                double step = m_eta0 / (t0 + t);
                TrainByOneSample(X[i],m_CD_K, step);
            }
            double cost = 0.0;
            for (int i = start; i < end; ++i)
            {
                cost += TestReconstructionError(X[i]);
            }
            Console.WriteLine("end evaluate eta:" + t0 + "\t construct error:" + cost);
            return cost;
        }

        public double DeterminInitT0(double[][] X, int start, int end)
        {

            double eta = m_eta0;
            double factor = 2.0;
            double lowT = eta;
            double highT = eta * factor;
            String file = "init_weight";
            SaveWeightsToFile(file);
            double lowCost = EvaluteCostByT0(X,start, end, lowT);
            ReadWeightsFromFile(file);
            double highCost = EvaluteCostByT0(X,start, end, highT);
            if (lowCost < highCost)
            {
                while (lowCost < highCost)
                {
                    highCost = lowCost;
                    highT = lowT;
                    lowT /= factor;
                    ReadWeightsFromFile(file);
                    lowCost = EvaluteCostByT0(X, start, end, lowT);
                }
            }
            else if (lowCost > highCost)
            {
                while (lowCost > highCost)
                {
                    lowCost = highCost;
                    lowT = highT;
                    highT *= factor;
                    ReadWeightsFromFile(file);
                    highCost = EvaluteCostByT0(X, start, end, highT);
                }
            }
            return lowT;
        }

        public void SaveWeightsToFile(String filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
            BinaryWriter bw = new BinaryWriter(fs);
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                for (int j = 0; j < m_HiddenDim; ++j)
                {
                    bw.Write(m_W[i, j]);
                }
            }
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                bw.Write(m_biasV[i]);
            }
            for (int i = 0; i < m_HiddenDim; ++i)
            {
                bw.Write(m_biasH[i]);
            }
            bw.Flush();
            bw.Close();
            fs.Close();
        }
        public void ReadWeightsFromFile(String filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read);
            BinaryReader br = new BinaryReader(fs);
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                for (int j = 0; j < m_HiddenDim; ++j)
                {
                    m_W[i, j]=br.ReadDouble();
                }
            }
            for (int i = 0; i < m_VisibleDim; ++i)
            {
                m_biasV[i] = br.ReadDouble();
            }
            for (int i = 0; i < m_HiddenDim; ++i)
            {
                m_biasH[i] = br.ReadDouble();
            }
            br.Close();
            fs.Close();
        }
    }
}

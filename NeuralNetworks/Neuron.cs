using System;
using System.Collections.Generic;

namespace NeuralNetworks
{
    public class Neuron
    {
        public List<double> Weight { get; }
        public NeuronType NeuType { get; }
        public double Output { get; private set; }
        public List<double> Inputs { get; private set; }
        public double Delta { get; private set; }
        public Neuron(int inputCount, NeuronType ttype = NeuronType.Normal)
        {
            NeuType = ttype;
            Weight = new List<double>();
            Inputs = new List<double>();

            InitWeightsRandomValues(inputCount);
        }

        private void InitWeightsRandomValues(int inputCount)
        {
            var rnd = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                if (NeuType != NeuronType.Input)
                {
                    Weight.Add(rnd.NextDouble());
                }
                else
                {
                    Weight.Add(1);
                }
               
                Inputs.Add(0.0);
            }
        }

        public void Learn(double erron, double learningRate)
        {
            if (NeuType == NeuronType.Input)
            {
                return;
            }
            Delta = erron * SigmoidDx(Output);

            for (int i = 0; i<Weight.Count; i++)
            {
                var weight = Weight[i];
                var input = Inputs[i];

                var nenWeight = weight - input * Delta * learningRate;
                Weight[i] = nenWeight;
            }

        }

        public double FeedForward(List<double> input)
        {
            for (int i = 0; i< Inputs.Count;i++)
            {
                Inputs[i] = input[i];
            }
            var sum = 0.0;
            for (int i = 0; i < input.Count; i++)
            {
                sum+= input[i]*Weight[i];
            }
            if (NeuType != NeuronType.Input)
            {
                Output = sigmoud(sum);
            }
            else Output = sum;
            return Output;
        }

        private double SigmoidDx(double x)
        {
            var sigmoid = sigmoud(x);
            var result = sigmoid / (1- sigmoid);
            return result;
        }

        private double sigmoud(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}

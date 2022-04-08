using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        public Topology Topology { get;  }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;    

            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayer();
            CreateOutputLayer();
        }
        public double Learn(List<Tuple<double, double[]>> dataset, int epoch)
        {
            var error = 0.0;
             for (int i = 0; i < epoch; i++)
            {
                foreach(var data in dataset)
                {
                    error += BackPropagetion(data.Item1,data.Item2);
                }
            }

            return error/epoch;
        }
        private double BackPropagetion(double expected, params double[] inputs)
        {
            var actual = FeedForvard(inputs).Output;

            var difference = actual - expected;

            foreach(var neuron in Layers.Last().NeuronList)
            {
                neuron.Learn(difference, Topology.LearningRayt);
            }

            for (int j = Layers.Count-2; j>=0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j+1];

                for (int i = 0; i<layer.NeuronList.Count; i++)
                {
                    var neuron = layer.NeuronList[i];
                    
                    for (int k = 0; k<previousLayer.NeurCount; k++)
                    {
                        var previosNeuron = previousLayer.NeuronList[k];
                        var error = previosNeuron.Weight[i] * previosNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRayt);
                    }
                    
                }
            }
            var result = difference * difference;
            return result;
        }
        public Neuron FeedForvard(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);

            FeedForvardAllLeyersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().NeuronList[0];
            }
            else
            {
                return Layers.Last().NeuronList.OrderByDescending(x => x.Output).FirstOrDefault();
            }
        }

        private void FeedForvardAllLeyersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var preLayersSignal = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.NeuronList)
                {
                    neuron.FeedForward(preLayersSignal);

                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var Signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].NeuronList[i];

                neuron.FeedForward(Signal);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neural = new Neuron(lastLayer.NeurCount, NeuronType.Output);
                outputNeurons.Add(neural);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayer()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var HiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neural = new Neuron(lastLayer.NeurCount);
                    HiddenNeurons.Add(neural);
                }
                var HiddenLayer = new Layer(HiddenNeurons, NeuronType.Output);
                Layers.Add(HiddenLayer);
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i<Topology.InputCount; i++)
            {
                var neural = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neural);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer); 
        }
    }
}

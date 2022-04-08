using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class Layer
    {
        public List<Neuron> NeuronList { get; set; }
        public int NeurCount => NeuronList?.Count ?? 0;
        public NeuronType Type;
        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            NeuronList = neurons;
            Type = type;
        }

        public List<double> GetSignals()
        {
            var result = new List<double>();   
            foreach(var neuron in NeuronList)
            {
                result.Add(neuron.Output);
            }
            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }

    }
}

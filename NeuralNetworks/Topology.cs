﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class Topology
    {
        public int InputCount { get; }
        public int OutputCount { get; }
        public double LearningRayt { get; }
        public List<int> HiddenLayers { get; }

        public Topology(int inputCount, int outputCount, double learningRayt, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRayt = learningRayt;
            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers);
        }
    }
}

#!/usr/bin/env python3
import visdom
import torch as t

if __name__ == "__main__":
    viz = visdom.Visdom(env="test")
    x = t.arange(1, 30, 0.01)
    y = t.sin(x)
    viz.line(X=x, Y=y, win="six2", opts={"title": "y=sin(x)"})

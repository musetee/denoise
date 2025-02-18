from torchviz import make_dot
#import os
#os.environ["PATH"] += os.pathsep + 'G:\projects\Graphviz-12.2.1-win64\bin'

def model_plot(model_class, input_sample):
    clf = model_class()
    y = clf(input_sample) 
    clf_view = make_dot(y, params=dict(list(clf.named_parameters()) + [('x', input_sample)]))
    return clf_view

# usage
# x = torch.randn(4, 1, 1376, 128).requires_grad_(True)
# model_plot(Classifier, x)
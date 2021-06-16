import  torch
from    torch import nn
from    torch.nn import functional as F
from    layer import GraphConvolution

from    config import args

class GCN(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)


        self.layers1 = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=True)
                                    ).to('cuda:0')
        self.layers2 = nn.Sequential(GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False)
                                    ).to('cuda:1')

    def forward(self, inputs):
        # x, support = inputs
        # x, support = self.layers1((x, support))
        # x = self.layers2((x, support))

        x, support = self.layers1(inputs)
        x = self.layers2((x.to('cuda:1'), support.to('cuda:1')))

        return x

    def l2_loss(self):

        layer = self.layers1.children()
        loss = None

        for l in layer:
            for p in l.parameters():
                if loss is None:
                    loss = p.pow(2).sum()
                else:
                    loss += p.pow(2).sum()
        loss = loss.to('cuda:1')
        # print(loss.device)
        layer = self.layers2.children()
        for l in layer:
            for p in l.parameters():
                if loss is None:
                    loss = p.pow(2).sum()
                else:
                    loss += p.pow(2).sum()
        
        return loss

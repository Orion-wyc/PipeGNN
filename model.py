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


        self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=True),

                                    GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False)

                                    )
                                    

    def forward(self, inputs):
        x, support = inputs

        x = self.layers((x, support))

        return x

    def l2_loss(self):
        # 为什么这里只算了一层的
        layer = self.layers.children()
        # print(type(layer))
        # layer = next(iter(layer)) # 这不就坑人么
        # print(type(layer))

        loss = None
        for l in layer:
            for p in l.parameters():
                # print(p.size())
                if loss is None:
                    loss = p.pow(2).sum()
                else:
                    loss += p.pow(2).sum()
            
        return loss


if __name__ == "__main__":
    net = GCN(2,3,3)
    _ = net.l2_loss()
    # layer = net.children()
    # print(type(net))
    # print(type(layer))
    # cnt = 0
    # for p in next(layer).parameters():
    #     cnt+=1
    #     print(cnt)
    #     print(p.size())
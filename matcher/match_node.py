import onnx 

class MatchNode():
    def __init__(self, node: onnx.NodeProto):
        self.build_node(node)
        
    def build_node(self, node: onnx.NodeProto):
        self.name = node.name
        self.type = node.op_type
        
        self.check_attr = False
        self.attr = {}
        
        if self.type.lower() in [
            "qlinearconv",
            "conv",
            "qlinearconvtranspose",
            "convtranspose"
        ]:
            self.check_attr = True
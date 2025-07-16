from ..layers.yolox.models.yolo_head import YOLOXHead

class CNNHead(YOLOXHead):
    def forward(self, xin):
        outputs = dict(cls_output=[], reg_output=[], obj_output=[])

        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            reg_feat = reg_conv(reg_x)

            outputs["cls_output"].append(self.cls_preds[k](cls_feat))
            outputs["reg_output"].append(self.reg_preds[k](reg_feat))
            outputs["obj_output"].append(self.obj_preds[k](reg_feat))

        return outputs

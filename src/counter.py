class ObjectCounter:
    def count(self,result,selcted_classes):
        class_count={class_name:0 for class_name in selcted_classes}

        for box in result.boxes:
            class_id=int(box.cls.item())
            class_name=result.names[class_id]
            if class_name in class_count:
                class_count[class_name]+=1

        total=sum(class_count.values())


        return {
            'count':class_count,
            'total':total
        }
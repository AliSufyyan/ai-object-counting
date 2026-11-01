import csv
from pathlib import Path 
class ObjectEvaluator:
    def __init__(self,
                 ground_truth_path='data/ground_truth.csv',
                 prediction_path='outputs/counting_result.csv'):
        self.ground_truth=Path(ground_truth_path)
        self.prediction_path=Path(prediction_path)


    def load_ground_truth(self):
        ground_truth={}
        with open(self.ground_truth, mode='r',newline='',encoding='utf-8') as file:
            reader=csv.DictReader(file)
            for row in reader:
                filename=row['filename']
                ground_truth[filename]={
                    class_name:int(count) for class_name,count in row.items() if class_name !='filename'
                }

        return ground_truth

    def load_prediction(self):
        prediction={}
        with open(self.prediction_path, mode='r',newline='',encoding='utf-8') as file:
            reader=csv.DictReader(file)
            for row in reader:
                filename=row['filename']
                class_name=row['class_name']
                count=int(row['count'])

                prediction.setdefault(filename,{})
                prediction[filename][class_name]=count 
        return prediction

    def evaluate(self):
        ground_truth=self.load_ground_truth()
        prediction=self.load_prediction()

        report=[]
        class_error={}
        for filename,actual_counts in ground_truth.items():
            if filename not in prediction:
                raise ValueError(f"Missing prediction for {filename}")

            for class_name,actual_count in actual_counts.items():
                if class_name not in prediction[filename]:
                    raise ValueError(f"Missing Prediction for {class_name} in {filename}")
                predicted_count=prediction[filename][class_name]
                absolute_error=abs(actual_count-predicted_count)

                class_error.setdefault(class_name,[])
                class_error[class_name].append(absolute_error)

                report.append({
                    'filename':filename,
                    'class_name':class_name,
                    'actual_count':actual_count,
                    'predicted_count':predicted_count,
                    'absolute_error':absolute_error
                })

        mae_result={
            class_name:sum(errors)/len(errors)
            for class_name,errors in class_error.items()
                                    }

        return report,mae_result

    def save_report(self,report,output_path='outputs/evaluation_report.csv'):
        output_dir=Path(output_path)
        output_dir.parent.mkdir(exist_ok=True,parents=True)

        fieldsname=[
            'filename',
            'class_name',
            'actual_count',
            'predicted_count',
            'absolute_error'
        ]

        with open(output_dir,mode='a',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=fieldsname)

            writer.writeheader()
            writer.writerows(report)

        return output_dir

            

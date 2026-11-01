from src.evaluator import ObjectEvaluator

evaluator=ObjectEvaluator()

report,mae_result=evaluator.evaluate()

for row in report:
    print(row)


print('\nMAE Result')
for class_name ,mae in mae_result.items():
    print(f'{class_name} : {mae:.2f}')

saved_path=evaluator.save_report(report)

print(f"\nEvaluation report saved: {saved_path}")
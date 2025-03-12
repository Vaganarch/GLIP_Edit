from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")


#results = model.val(data="../datasets/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/aquarium.yaml")
#results = model.val(data="../datasets/coco8/coco8.yaml")
#results = model.val(data="../datasets/coco/coco2017.yaml")
results = model.val(data="../datasets/PennFudanPed/penn_yolo.yaml")

print("Class indices with average precision:", results.ap_class_index)
print("Average precision for all classes:", results.box.all_ap)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean recall:", results.box.mr)
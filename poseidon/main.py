from generation.coco_instance_generator import COCOInstanceGenerator
from extraction.coco_instance_extractor import COCOInstanceExtractor
from utils.normalization import COCONormalization
from utils.coco2yolo import COCO2YOLO

if __name__ == '__main__':
    extractor = COCOInstanceExtractor()
    extractor.dataset_stats()
    extractor.extract()

    generator = COCOInstanceGenerator()
    generator.balance('/Users/pabloruizponce/Vainas/POSEIDON/poseidon/outputs')
    
    #conversor = COCO2YOLO()
    #conversor.convert("./SDSYOLO", "SDSYOLO"

    #normalizator = COCONormalization()
    #normalizator.normalize("./SDSYOLO")


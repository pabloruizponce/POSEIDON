from generation.coco_instance_generator import COCOInstanceGenerator
from extraction.coco_instance_extractor import COCOInstanceExtractor

if __name__ == '__main__':
    extractor = COCOInstanceExtractor()
    extractor.dataset_stats()
    extractor.extract()

    generator = COCOInstanceGenerator()
    generator.balance('/Users/pabloruizponce/Vainas/POSEIDON/poseidon/outputs')
    

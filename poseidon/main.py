from extraction.coco_instance_extractor import COCOInstanceExtractor

if __name__ == '__main__':
    extractor = COCOInstanceExtractor()
    extractor.dataset_stats()
    extractor.extract()


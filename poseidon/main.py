from extraction.coco_instance_extractor import COCOInstanceExtractor

if __name__ == '__main__':
    extractor = COCOInstanceExtractor()
    extractor.dataset_stats()
    extractor.extract()
    extractor.extract_background(background_size=(1000,1000), stride=(100,100))


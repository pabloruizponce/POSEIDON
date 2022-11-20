from extraction.coco_instance_extractor import COCOInstanceExtractor

if __name__ == '__main__':
    extractor = COCOInstanceExtractor()
    print(extractor.base_path)
    print(extractor.val_annotations['categories'])


from steps.etl import DataExtractor, DataTransform, DataLoader, DataManipulations

input = "sample-brokers.txt"
output = "output-sample.csv"

filters = ["costa del este", "paitilla","amoblado","venta", "2023", "3"]
sort_order = ['year','month','day','contact']
order_data = 'asc'


data_extractor = DataExtractor(file_path=input)
data_extractor.read_file()

data_transformer = DataTransform(data=data_extractor.raw_data)
data_transformer.split_data()
data_transformer.clear_messages()
data_transformer.fuzzy_match()


# data_filter = DataManipulations(filters=filters, sort_order=order_data )
# data_filter.apply_filters(data_transformer.data_transform)
# data_filter.sort_data(data_filter.data)

data_loader = DataLoader(output, data_transformer.data_transform)
data_loader.save_file()
data_loader.print_data()



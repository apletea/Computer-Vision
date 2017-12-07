
	if(argc<5){
		cout<<"Requires more arguments.\n"
		<<"Correct format: ./genXORtraindata model.prototxt trainedmodel.caffemodel inputDataLayerName outputName\n";
		return(-1);
	}
	
	// caffe is using google logging (aka 'glog') as its logging module, and hence this module must be initialized once when running caffe. Therefore the following line 
	::google::InitGoogleLogging(argv[0]);
	
	// load the trained weights cached inside XOR_iter_5000000.caffemodel
	shared_ptr<Net<float> >	testnet;
	
	testnet.reset(new Net<float>(argv[1],TEST));
	testnet->CopyTrainedLayersFrom(argv[2]);
	
	// obtain the input MemoryData layer and pass the input to it for testing
	float testab[] = {0, 0, 0 , 1, 1, 0, 1, 1};
	float testc[] = {0, 1, 1, 0};
	
	MemoryDataLayer<float> *dataLayer_testnet = 
	(MemoryDataLayer<float> *)
	(testnet->layer_by_name(argv[3]).get());
	
	dataLayer_testnet->Reset(testab, testc, 4);
	
	// calculate the neural network output
	testnet->Forward();
	
	// access blobs to display results
	boost::shared_ptr<Blob<float> > output_layer =
	testnet->blob_by_name(argv[4]);
	
	const float* begin = output_layer->cpu_data();
	const float* end = begin + 4;
	
	// We know the output size is 4, and we save the outputs into the result vector
	vector<float> result(begin, end);
	
	// display the result
	for(int i=0; i<result.size(); i++){
		cout<<"input: "<<(int)testab[i*2 + 0]
		<<" xor "<<(int)testab[i*2 + 1]
		<<", truth: "<<(int)testc[i]
		<<", result by NN: "<<result[i]<<endl;
	}
	

return(0);

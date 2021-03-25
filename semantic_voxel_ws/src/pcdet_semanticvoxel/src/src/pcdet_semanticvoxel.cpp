// Original packages refer to /usr/Desktop/pcdet_semanticvoxel_cpp
#include "pcdet_semanticvoxel.h"


Pcdet_semanticvoxel::Pcdet_semanticvoxel(const char *input_topic,const float score_thd_input, const float nms_overlap_thresh_input,
		   const float voxel_size_x, const float voxel_size_y, const float voxel_size_z, 
		   const float coors_range_xmin, const float coors_range_xmax,
		   const float coors_range_ymin, const float coors_range_ymax, 
		   const float coors_range_zmin, const float coors_range_zmax,const char *module_loc,const char *module_before_nms_loc,const char *backbone_loc)
{
	test = true;
	paintvoxel_reverse_ptr_.reset(new PaintVoxel());
	// set ros handle
	// set a publisher for detection
	detectionsPublisher = nh.advertise<vision_msgs::Detection3DArray>("detection_publisher",1);
	
	// set ROS rate
	
	
	ROS_INFO("ROS ready!");
	topic = input_topic;
    //get_filelist_from_dir(bin_root, bin_files);
    // init start
	
    try
    {
        module = torch::jit::load(module_loc);

        module.to(torch::kCUDA);
        module_before_nms = torch::jit::load(module_before_nms_loc);
        module_before_nms.to(torch::kCUDA);


		
    }
    catch (const c10::Error &e)
    {
        //std::cerr << "error loading the model " << e.what() << std::endl;
        throw "error loading the model ";
        //return -1;
    }

    ret = backbone.initEngine(backbone_loc);
    if (ret != ALGORITHM_OPERATION_SUCCESS)
    {
    	throw "Algorithm Operation Failed";
        //return -1;
    }
    redo = false;

    scatter_cuda_ptr_.reset(new ScatterCuda(in_channels, nx*ny));

    score_thd = score_thd_input;
    nms_overlap_thresh = nms_overlap_thresh_input;

	voxel_size.push_back(voxel_size_x);
	voxel_size.push_back(voxel_size_y);
	voxel_size.push_back(voxel_size_z);


	coors_range.push_back(coors_range_xmin);
	coors_range.push_back(coors_range_xmax);
	coors_range.push_back(coors_range_ymin);
	coors_range.push_back(coors_range_ymax);
	coors_range.push_back(coors_range_zmin);
	coors_range.push_back(coors_range_zmax);
	timestart =ros::Time::now();    

}
void Pcdet_semanticvoxel::doEveryThing(void)
{

	pointCloudSubscriber = nh.subscribe<pcdet_semanticvoxel::matrix2D_msg>(topic,1,boost::bind(&Pcdet_semanticvoxel::pointsCloudSubCallback,this,_1));
	//ros::spin();

}
void Pcdet_semanticvoxel::publishDetectionArray(const at::Tensor final_boxes, const at::Tensor final_labels, const at::Tensor final_scores, const pcdet_semanticvoxel::matrix2D_msgConstPtr& pCloud)
{
// publish the detection
		vision_msgs::Detection3DArray detectionsMessages;
		/*
		pcl::PointCloud<pcl::PointXYZI> cloud;
		sensor_msgs::PointCloud2 cloudMsg;
		cloud.width = 1;
		cloud.height = points.size(0);
		cloud.points.resize(cloud.width*cloud.height);
		std::cout<<points<<std::endl;
		for(int i=0;i<points.size(0);i++)
		{
			//std::cout<<points[i]<<std::endl;
			cloud.points[i].x = points[i][0].item<float>();
			cloud.points[i].y = points[i][1].item<float>();
			cloud.points[i].z = points[i][2].item<float>();
			cloud.points[i].intensity = points[i][3].item<float>();
			//for(int j=0;j<points[i].size(2);j++)
			//{
				//std::cout<<points[i][j].item<float>()<<std::endl;
			//}
		}
		//pcl::toROSMsg(cloud, cloudMsg);
		//cloudMsg.header.frame_id = "velo_link";
		*/

		for(int i=0;i<final_scores.size(0);i++)
		{
			
			if(final_scores[i].item<float>() < score_thd) continue;
			std::cout<<score_thd<<std::endl;
			std::cout<<final_scores[i].item<float>()<<std::endl;
			vision_msgs::Detection3D msg;
			vision_msgs::ObjectHypothesisWithPose hyp;
			//msg.bbox.results.id = i;
			//msg.bbox.results.score = final_scores[i].item<float>();
			msg.bbox.center.position.x = final_boxes[0][i][0].item<float>();
			msg.bbox.center.position.y = final_boxes[0][i][1].item<float>();
			msg.bbox.center.position.z = final_boxes[0][i][2].item<float>();
			msg.bbox.center.orientation.x = 0;
			msg.bbox.center.orientation.y = 0;
			msg.bbox.center.orientation.z = 0;
			msg.bbox.center.orientation.w = final_boxes[0][i][6].item<float>();
			msg.bbox.size.x = final_boxes[0][i][3].item<float>();
			msg.bbox.size.y = final_boxes[0][i][4].item<float>();
			msg.bbox.size.z = final_boxes[0][i][5].item<float>();
			hyp.id = (int) final_labels[i].item<long>();
			hyp.score = final_scores[i].item<float>();
			msg.results.push_back(hyp);
			
			detectionsMessages.detections.push_back(msg);
			//std::cout<<final_scores[i].item<float>()<<std::endl;
			//std::cout<<final_labels[i].item<long>()<<std::endl;
			//std::cout<<final_boxes[0][i][0].item<float>()<<std::endl;
		}
		detectionsMessages.header.stamp = pCloud->header.stamp;
		detectionsMessages.header.frame_id = pCloud->header.frame_id;
		detectionsPublisher.publish(detectionsMessages);
}
void Pcdet_semanticvoxel::pointsCloudSubCallback(const pcdet_semanticvoxel::matrix2D_msgConstPtr& pCloud)
{
	std::cout << (ros::Time::now()-timestart) << std::endl;
	std::cout << pCloud->x_size << pCloud->y_size << std::endl;
	float point_array[pCloud->x_size*pCloud->y_size];
	auto options = torch::TensorOptions().dtype(torch::kFloat);
	memcpy(&point_array,&(pCloud->data[0]),pCloud->x_size*pCloud->y_size*sizeof(float));
 	torch::Tensor points = torch::from_blob(point_array,{pCloud->x_size,pCloud->y_size},options);
    torch::Tensor batch_cls_preds = torch::zeros({1, 321408, 3}, at::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();
    torch::Tensor batch_box_preds = torch::zeros({1, 321408, 7}, at::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();
    torch::Tensor dir_labels = torch::zeros({1, 321408}, at::requires_grad(false).dtype(at::kInt)).cuda().contiguous();
	points = points.cuda();
	
    auto spatial_feature = torch::zeros({in_channels, nx * ny}, torch::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();
    int size_1 = pCloud->x_size * pCloud->y_size;
	auto voxels = torch::zeros({max_voxels, max_points, points.size(1)}, torch::requires_grad(false).dtype(torch::kFloat)).cuda().contiguous();
        auto coors = torch::zeros({max_voxels, 3}, torch::requires_grad(false).dtype(torch::kInt)).cuda().contiguous();
        auto num_points_per_voxel = torch::zeros(max_voxels, torch::requires_grad(false).dtype(torch::kInt)).cuda().contiguous();
        auto start_total = std::chrono::steady_clock::now();
        auto start = std::chrono::steady_clock::now();

        // 2. voxelization
        int voxel_num = voxelization::hard_voxelize(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3);
        auto end = std::chrono::steady_clock::now();
        std::cout << "hard_voxelize time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        std::cout<<"pillar num = "<<voxel_num<<std::endl;
        // voxelization结果正常，差值为0

        // 3. paint_voxel
        auto bev_map = torch::zeros({1, 24, 496, 432}, at::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();
        paintvoxel_reverse_ptr_->paint_voxel_reverse(voxel_num, voxels, num_points_per_voxel, coors, voxel_size, coors_range, bev_map);
//        std::cout<<bev_map[0][21][0][248]<<std::endl;
        // bev_map 结果异常，差值为0，paint_voxel实现异常
        /**
        [-1.         -0.875      -0.85714287 -0.8333333  -0.8        -0.7777778
         -0.75       -0.71428573 -0.7        -0.6666667  -0.6363636  -0.625
         -0.5        -0.42857143 -0.4        -0.375      -0.33333334 -0.2857143
         -0.25       -0.22222222 -0.2        -0.16666667 -0.15789473 -0.14285715
         -0.07692308  0.          0.25        0.5         0.6666667   1.        ]
        * */
        // receive bytes in a `std::vector<char>`
//        torch::Tensor bev_map_python=loadTensor("/home/shining/work/Projects/work/python/tools/python_result/bev_map.zip");
//        bev_map_python.to(torch::kCUDA);

        torch::Tensor zero_tensor = torch::zeros({max_voxels, 1}, torch::requires_grad(false).dtype(at::kInt)).cuda();
        coors = torch::cat({zero_tensor, coors}, 1); //
        start = std::chrono::steady_clock::now();
        std::vector<torch::jit::IValue> inputs = {voxels, num_points_per_voxel, coors};
        /**
         * vfe选用trace部署：因为trt对vfe中的激活函数不友好，加速之后变得很慢
         * */
        // 4. vfe

        auto ans = module.forward(inputs).toTensor();
        end = std::chrono::steady_clock::now();
        std::cout << "vfe_pfn time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        // vfe 结果正常，差值为0
        auto voxel_features = ans.slice(0, 0, voxel_num).squeeze(1); //.squeeze(0).squeeze(1)
        auto _coors = coors.slice(0, 0, voxel_num);


        // middle encoder start
        // Create the canvas for this sample
        // 5. PointPillarScatter:middle_encoder
        start = std::chrono::steady_clock::now();
        auto this_coors = _coors;
        torch::Tensor t_indices = this_coors.slice(1, 1, 2, 1) + this_coors.slice(1, 2, 3, 1) * nx + this_coors.slice(1, 3, 4, 1); // [xxx, 1]
        auto indices = t_indices.toType(at::kLong).view(-1).contiguous();
        auto temp_voxels = voxel_features.t().contiguous(); // temp_voxels: [64, xxxx]
        /**
         * cuda函数实现index_put操作，因为当前版本的torch，index_put函数有内存溢出
         * */
        scatter_cuda_ptr_->doScatterCuda(temp_voxels.data_ptr<float>(), indices.data_ptr<long>(), indices.size(0), spatial_feature.data_ptr<float>());
        end = std::chrono::steady_clock::now();
        std::cout << "cuda scatter time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        // Undo the column stacking to final 4-dim tensor
        torch::Tensor spatial_features = spatial_feature.view({1, in_channels, ny, nx}).contiguous();
        // PointPillarScatter 结果正常，差值为0

        // backbone start
        start = std::chrono::steady_clock::now();
        cudaMemset(batch_cls_preds.data_ptr(), 0, 321408 * 3 * sizeof(float));
        cudaMemset(batch_box_preds.data_ptr(), 0, 321408 * 7 * sizeof(float));
        cudaMemset(dir_labels.data_ptr(), 0, 321408 * sizeof(float));
        end = std::chrono::steady_clock::now();
        std::cout << "tensor alcocate time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;


        // 6. backbone_2d + AnchorHeadSingle
        vector<void *> buffers = {spatial_features.data_ptr(),
                                  bev_map.data_ptr(),
                                  batch_cls_preds.data_ptr(),
                                  batch_box_preds.data_ptr(),
                                  dir_labels.data_ptr()};
        start = std::chrono::steady_clock::now();
        /**
         * tensorrt backend
         * */
        ret = backbone.infer(buffers, 1);
        end = std::chrono::steady_clock::now();
        std::cout << "backbone time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        if (ret != ALGORITHM_OPERATION_SUCCESS)
        {
            return;
        }




        //7. nms 前处理
        // limit period
        auto val = batch_box_preds.slice(2, 6, 7, 1).squeeze(2) - dir_offset;
        auto dir_rot = val - torch::floor(val / period) * period;
        batch_box_preds = torch::cat({batch_box_preds.slice(2, 0, 6), (dir_rot + dir_offset + period * dir_labels).unsqueeze_(2)}, 2).contiguous();

        start = std::chrono::steady_clock::now();



        /**
         * trace：好处少写代码
         * */
        std::vector<torch::jit::IValue> inputs_1 = {batch_box_preds, batch_cls_preds};
        auto before_nms = module_before_nms.forward(inputs_1).toTuple();
        auto cls_preds = before_nms->elements()[0].toTensor();
        auto label_preds = before_nms->elements()[1].toTensor();
        auto scores_mask = before_nms->elements()[2].toTensor();
        auto boxes_for_nms = before_nms->elements()[3].toTensor().contiguous();
        auto order = before_nms->elements()[4].toTensor();
        auto indices_topk = before_nms->elements()[5].toTensor();
        torch::Tensor keep = torch::zeros({boxes_for_nms.size(0)}, at::requires_grad(false).dtype(at::kLong)).cpu().contiguous();
        int64_t index = 0;
        int num_out = 0;
        if( redo)
        {
            // auto cpu_indices = indices.cpu();
            // int id_0 = cpu_indices.data_ptr<long>()[0];
            // int id_1 = cpu_indices.data_ptr<long>()[1];
            // std::cout<<spatial_feature[0][id_0]<<std::endl;
            // std::cout<<spatial_feature[0][id_1]<<std::endl;

            std::cout<<batch_box_preds[0][0]<<std::endl;
            std::cout<<batch_box_preds[0][1]<<std::endl;
            std::cout<<batch_cls_preds[0][0]<<std::endl;
            std::cout<<batch_cls_preds[0][1]<<std::endl;
            redo = false;
        }
        //8. nms
        if(boxes_for_nms.size(0) > 0)
        {
            num_out = nms_gpu(boxes_for_nms, keep, 0.01);
        }
        else
        {
            std::cout<<"zero predict"<<std::endl;

        }
        keep = keep.slice(0,0,num_out).cuda();
        auto keep_idx = order.index_select(0, keep);
        if(keep_idx.size(0) > NMS_POST_MAXSIZE)
            keep_idx = keep_idx.slice(0, 0 , NMS_POST_MAXSIZE);
        auto selected = indices_topk.index_select(0,keep_idx);
        auto original_idxs = scores_mask.nonzero().view(-1);
        selected = original_idxs.index_select(0, selected);
        auto final_scores = cls_preds.index_select(0, selected);
        auto final_labels = label_preds.index_select(0, selected);
        auto final_boxes = batch_box_preds.index_select(1, selected);
        std::cout << "nms time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        std::cout << "total time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start_total).count() / 1000 << "ms" << std::endl;
		publishDetectionArray(final_boxes,final_labels,final_scores, pCloud);
		if(final_boxes.size(1) != final_labels.size(0) || final_scores.size(0) != final_boxes.size(1) || final_labels.size(0) != final_scores.size(0)) {
			std::cout<<final_boxes.size(1)<<final_labels.size(0)<<final_scores.size(0)<<std::endl;
		}
		

}




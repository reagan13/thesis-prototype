import React from "react";
import CustomTable from "../components/CustomTable";

const Page = ({ name, categoryResponse, text, intentResponse, isSidebarCollapsed }) => {
	console.log("Page", categoryResponse, intentResponse);
	return (
		<div className={" h-[610px] p-6  bg-[#133075d2] rounded-lg shadow-lg transition-transform transform hover:scale-105 hover:shadow-xl outline outline-2 outline-white outline-offset-2"}
		style={{
			width: isSidebarCollapsed ? "1250px" : "1150px",
			transition: "all 0.3s ease",
		}}>
		
			<div className="text-2xl font-bold mb-2 text-white text-center pb-[20px]">{name}</div>
			<div className="grid grid-cols-2 gap-4 mb-4">
				<div className="bg-[#CAF0F8] rounded-lg shadow-lg outline outline-2 outline-white outline-offset-2 p-4">
					<h2 className="text-lg font-semibold text-gray-800 border-b-2 border-blue-500 pb-1">
						Multi Task Classifications
					</h2>
					<div className="flex items-center justify-between mt-2 ">
						<span className="text-gray-700 ">NER</span>
						<span className="text-gray-700 ">Unknown</span>
					</div>
					<div className="flex items-center justify-between mt-1">
						<span className="text-gray-700">Category</span>
						<span className="text-gray-700">{categoryResponse.data.class}</span>
					</div>
					<div className="flex items-center justify-between mt-1">
						<span className="text-gray-700">Intent</span>
						<span className="text-gray-700">{intentResponse.data.class}</span>
					</div>
				</div>

				<div className="bg-[#CAF0F8] rounded-lg shadow-lg outline outline-2 outline-white outline-offset-2 p-4">
					<h2 className="text-lg font-semibold text-gray-800  pb-1 ">
						Analysis
					</h2>
					<CustomTable
						categoryResponse={categoryResponse.data.probabilities}
						intentResponse={intentResponse.data.probabilities}
					/>
				</div>
			</div>

			<div>
				<h2 className="pt-[4px] text-lg font-semibold text-white border-b-2 border-blue-500">
					Graph
				</h2>
				<p className="text-white mt-2 h-[270px] ">Graph Area</p>
			</div>
			{/* <div>
				<h2 className="text-lg font-semibold text-gray-800 border-b-2 border-blue-500 pb-1">
					Response
				</h2>
				<p className="text-gray-700 mt-2">
					The predicted intent is {intentResponse.data.class}
				</p>
				<p className="text-gray-700 mt-2">
					the predicted category is {categoryResponse.data.class}
				</p>
			</div> */}
		</div>
		
	);
};

export default Page;

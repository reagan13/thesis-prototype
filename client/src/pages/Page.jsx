import React from "react";
import PropTypes from "prop-types";
import CustomTable from "../components/CustomTable";
const Page = ({ name, response, categorization, analysis }) => {
	return (
		<div className="p-6 bg-white rounded-lg shadow-lg transition-transform transform hover:scale-105 hover:shadow-xl border border-gray-200">
			<div className="text-2xl font-bold mb-2 text-blue-600 text-center">
				{name}
			</div>
			<div className="mb-4">
				<h2 className="text-lg font-semibold text-gray-800 border-b-2 border-blue-500 pb-1">
					Multi Task Classifications
				</h2>
				<div className="flex items-center justify-between mt-2">
					<span className="text-gray-700">NER</span>
					<span className="text-gray-700">70%</span>
				</div>
				<div className="flex items-center justify-between mt-1">
					<span className="text-gray-700">Category</span>
					<span className="text-gray-700">20%</span>
				</div>
				<div className="flex items-center justify-between mt-1">
					<span className="text-gray-700">Intent</span>
					<span className="text-gray-700">10%</span>
				</div>
			</div>
			<div className="mb-4">
				<h2 className="text-lg font-semibold text-gray-800 border-b-2 border-blue-500 pb-1">
					Analysis
				</h2>
				<CustomTable />
			</div>
			<div>
				<h2 className="text-lg font-semibold text-gray-800 border-b-2 border-blue-500 pb-1">
					Response
				</h2>
				<p className="text-gray-700 mt-2">{response}</p>
			</div>
		</div>
	);
};

export default Page;

Page.propTypes = {
	name: PropTypes.string,
	response: PropTypes.string,
	categorization: PropTypes.string,
	analysis: PropTypes.string,
};

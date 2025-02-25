// components/BarChart.jsx
import React from "react";

const BarChart = ({ title, description, color, onClick }) => {
	return (
		<div
			className="p-6 bg-white border border-gray-200 rounded-lg shadow-md cursor-pointer hover:shadow-lg transition duration-300"
			style={{ backgroundColor: `${color}20` }} // Add transparency to the background color
			onClick={onClick}
		>
			<h3 className="text-xl font-bold text-gray-800">{title}</h3>
			<p className="text-gray-600 mt-2">{description}</p>
		</div>
	);
};

export default BarChart;

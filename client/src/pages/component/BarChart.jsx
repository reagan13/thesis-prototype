import React, { useEffect, useRef } from "react";
import { Bar } from "react-chartjs-2";
import {
	Chart as ChartJS,
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
	Tooltip,
	Legend,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
	Tooltip,
	Legend
);

const ChartBox = ({ title, description, onClick }) => {
	const chartRef = useRef(null); // Reference to the chart instance

	// Example data for the bar chart
	const chartData = {
		labels: ["Cancel Order", "Change Order"], // Detected classes as x-axis labels
		values: [40, 60], // Confidence scores (percentage values)
	};

	useEffect(() => {
		// Cleanup function to destroy the chart when the component unmounts or re-renders
		return () => {
			if (chartRef.current) {
				chartRef.current.destroy(); // Destroy the previous chart instance
			}
		};
	}, []);

	return (
		<div className="border border-black w-[400px] h-[250px]">
			<Bar
				ref={chartRef} // Assign the ref to the chart instance
				data={{
					labels: chartData.labels, // X-axis labels
					datasets: [
						{
							label: "Confidence Score", // Label for the
							data: chartData.values, // Both Cancel Order and Change Order values
							backgroundColor: [
								"rgba(43, 63, 229, 0.8)", // Blue for Cancel Order (Baseline)
								"rgba(250, 192, 19, 0.8)", // Orange for Change Order (Hybrid)
							],
							borderColor: [
								"rgba(43, 63, 229, 1)", // Border color for Cancel Order
								"rgba(250, 192, 19, 1)", // Border color for Change Order
							],
							borderWidth: 1,
							borderRadius: 5,
						},
					],
				}}
				options={{
					plugins: {
						title: {
							display: true,
							text: title || "Revenue Source", // Use the title prop if provided
						},
						legend: {
							display: true, // Ensure the legend is displayed
							position: "top", // Position the legend at the top
							labels: {
								boxWidth: 20, // Adjust the size of the legend color boxes
								padding: 10, // Add padding between legend items
							},
						},
					},
					responsive: true, // Ensure the chart is responsive
					maintainAspectRatio: false, // Adjust aspect ratio
					scales: {
						y: {
							beginAtZero: true, // Start the y-axis at zero
							max: 100, // Set the maximum value of the y-axis to 100 (for percentage)
							ticks: {
								callback: (value) => `${value}%`, // Show percentage on y-axis
							},
						},
					},
				}}
			/>
		</div>
	);
};

export default ChartBox;

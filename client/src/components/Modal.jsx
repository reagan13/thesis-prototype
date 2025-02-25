// components/Modal.jsx
import PropTypes from "prop-types";
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
import { X } from "lucide-react"; // Import the X icon from Lucide React
import { useEffect, useState } from "react";

// Register Chart.js components
ChartJS.register(
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
	Tooltip,
	Legend
);

const Modal = ({ selectedBox, onClose }) => {
	const [chartData, setChartData] = useState({ baseline: [], hybrid: [] });

	// Fetch chart data dynamically (mocked here)
	useEffect(() => {
		const fetchChartData = async () => {
			// Replace this with actual API call or context data
			const mockData = {
				Intent: {
					baseline: {
						labels: ["Cancel Order", "Change Order"],
						values: [40, 60],
					},
					hybrid: {
						labels: ["Cancel Order", "Change Order"],
						values: [50, 50],
					},
				},
				Category: {
					baseline: { labels: ["Support", "Feedback"], values: [30, 70] },
					hybrid: { labels: ["Support", "Feedback"], values: [60, 40] },
				},
				NER: {
					baseline: [
						{ entity: "Person", confidence: 0.9 },
						{ entity: "Order ID", confidence: 0.8 },
						{ entity: "Location", confidence: 0.7 },
					],
					hybrid: [
						{ entity: "Person", confidence: 0.85 },
						{ entity: "Order ID", confidence: 0.8 },
						{ entity: "Location", confidence: 0.75 },
					],
				},
			};

			// Ensure proper initialization for NER
			const selectedChartData = mockData[selectedBox] || {
				baseline: [],
				hybrid: [],
			};
			setChartData({
				baseline: Array.isArray(selectedChartData.baseline)
					? selectedChartData.baseline.slice(0, 5) // Limit to top 5 entities
					: [],
				hybrid: Array.isArray(selectedChartData.hybrid)
					? selectedChartData.hybrid.slice(0, 5) // Limit to top 5 entities
					: [],
			});
		};
		fetchChartData();
	}, [selectedBox]);

	return (
		<div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
			{/* Modal Content */}
			<div className="bg-white rounded-lg max-w-6xl p-6 relative w-[1300px] h-[650px] overflow-hidden">
				{/* Header Section */}
				<div className="flex justify-between items-center mb-6">
					<h2 className="text-xl font-bold">{selectedBox} Analysis</h2>
					<button
						className="text-gray-500 hover:text-gray-700 transition-colors"
						onClick={onClose}
						aria-label="Close modal"
					>
						<X size={24} /> {/* Lucide React X Icon */}
					</button>
				</div>
				{/* Main Content */}
				<div className="flex border border-black w-full h-[550px] overflow-hidden">
					{/* Left Side: Baseline Model */}
					<div className="w-1/2 pr-4 overflow-hidden">
						<h3 className="text-lg font-semibold mb-4">Baseline Model</h3>
						{["Intent", "Category"].includes(selectedBox) && (
							<div className="h-[500px] overflow-hidden">
								<Bar
									data={{
										labels: chartData.baseline?.labels || [],
										datasets: [
											{
												label: "Confidence Score (%)",
												data: chartData.baseline?.values || [],
												backgroundColor: "rgba(43, 63, 229, 0.8)",
												borderColor: "rgba(43, 63, 229, 1)",
												borderWidth: 1,
												borderRadius: 5,
											},
										],
									}}
									options={{
										indexAxis: "y", // Horizontal bar chart
										plugins: {
											title: {
												display: false,
											},
											legend: {
												display: true,
												position: "top",
											},
										},
										responsive: true,
										maintainAspectRatio: false, // Ensure the chart scales properly
										scales: {
											y: {
												beginAtZero: true,
												grid: {
													display: false, // Remove grid lines for a cleaner look
												},
											},
											x: {
												beginAtZero: true,
												max: 100,
												ticks: {
													stepSize: 20, // Set step size to 20
													callback: (value) => `${value}%`, // Append "%" to each tick
												},
											},
										},
									}}
								/>
							</div>
						)}
						{selectedBox === "NER" && (
							<div className="h-[500px] overflow-y-auto">
								{chartData.baseline.length > 0 ? (
									<Bar
										data={{
											labels: chartData.baseline.map((item) => item.entity),
											datasets: [
												{
													label: "Confidence Score (%)",
													data: chartData.baseline.map(
														(item) => item.confidence * 100
													),
													backgroundColor: "rgba(43, 63, 229, 0.8)",
													borderColor: "rgba(43, 63, 229, 1)",
													borderWidth: 1,
													borderRadius: 5,
												},
											],
										}}
										options={{
											indexAxis: "y", // Horizontal bar chart
											plugins: {
												title: {
													display: false,
												},
												legend: {
													display: true,
													position: "top",
												},
											},
											responsive: true,
											maintainAspectRatio: false, // Ensure the chart scales properly
											scales: {
												y: {
													beginAtZero: true,
													grid: {
														display: false, // Remove grid lines for a cleaner look
													},
												},
												x: {
													beginAtZero: true,
													max: 100,
													ticks: {
														stepSize: 20, // Set step size to 20
														callback: (value) => `${value}%`, // Append "%" to each tick
													},
												},
											},
										}}
									/>
								) : (
									<p className="text-sm text-gray-700">
										No named entities detected.
									</p>
								)}
							</div>
						)}
					</div>

					{/* Right Side: Hybrid Model */}
					<div className="w-1/2 pl-4 overflow-hidden">
						<h3 className="text-lg font-semibold mb-4">Hybrid Model</h3>
						{["Intent", "Category"].includes(selectedBox) && (
							<div className="h-[500px] overflow-hidden">
								<Bar
									data={{
										labels: chartData.hybrid?.labels || [],
										datasets: [
											{
												label: "Confidence Score (%)",
												data: chartData.hybrid?.values || [],
												backgroundColor: "rgba(250, 192, 19, 0.8)",
												borderColor: "rgba(250, 192, 19, 1)",
												borderWidth: 1,
												borderRadius: 5,
											},
										],
									}}
									options={{
										indexAxis: "y", // Horizontal bar chart
										plugins: {
											title: {
												display: false,
											},
											legend: {
												display: true,
												position: "top",
											},
										},
										responsive: true,
										maintainAspectRatio: false, // Ensure the chart scales properly
										scales: {
											y: {
												beginAtZero: true,
												grid: {
													display: false, // Remove grid lines for a cleaner look
												},
											},
											x: {
												beginAtZero: true,
												max: 100,
												ticks: {
													stepSize: 20, // Set step size to 20
													callback: (value) => `${value}%`, // Append "%" to each tick
												},
											},
										},
									}}
								/>
							</div>
						)}
						{selectedBox === "NER" && (
							<div className="h-[500px] overflow-y-auto">
								{chartData.hybrid.length > 0 ? (
									<Bar
										data={{
											labels: chartData.hybrid.map((item) => item.entity),
											datasets: [
												{
													label: "Confidence Score (%)",
													data: chartData.hybrid.map(
														(item) => item.confidence * 100
													),
													backgroundColor: "rgba(250, 192, 19, 0.8)",
													borderColor: "rgba(250, 192, 19, 1)",
													borderWidth: 1,
													borderRadius: 5,
												},
											],
										}}
										options={{
											indexAxis: "y", // Horizontal bar chart
											plugins: {
												title: {
													display: false,
												},
												legend: {
													display: true,
													position: "top",
												},
											},
											responsive: true,
											maintainAspectRatio: false, // Ensure the chart scales properly
											scales: {
												y: {
													beginAtZero: true,
													grid: {
														display: false, // Remove grid lines for a cleaner look
													},
												},
												x: {
													beginAtZero: true,
													max: 100,
													ticks: {
														stepSize: 20, // Set step size to 20
														callback: (value) => `${value}%`, // Append "%" to each tick
													},
												},
											},
										}}
									/>
								) : (
									<p className="text-sm text-gray-700">
										No named entities detected.
									</p>
								)}
							</div>
						)}
					</div>
				</div>
			</div>
		</div>
	);
};

Modal.propTypes = {
	selectedBox: PropTypes.string.isRequired,
	onClose: PropTypes.func.isRequired,
};

export default Modal;

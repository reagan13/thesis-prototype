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
import { useData } from "../../context/DataContext"; // Import useData

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
	const { selectedBotResponse, data } = useData(); // Get selectedBotResponse and data from context
	const messages = data.messages || [];

	// Find the user input corresponding to the selected bot response
	const userInput = messages.find(
		(message) =>
			message.botResponses &&
			message.botResponses.some((response) => response.id === selectedBotResponse?.id)
	)?.text || "No user input found";

	// Example data for the bar chart
	const chartData = {
		Intent: { labels: ["Cancel Order", "Change Order"], values: [40, 60] },
		Category: { labels: ["Cancel Order", "Change Order"], values: [40, 60] },
		NER: { labels: ["Cancel Order", "Change Order"], values: [40, 60] },
	};

	// Get the data for the selected box
	const { labels, values } = chartData[selectedBox] || { labels: [], values: [] };

	return (
		<div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
			{/* Modal Content */}
			<div className="bg-white rounded-lg max-w-6xl p-6 relative w-[1300px] h-[650px]">
				{/* Header Section */}
				<div className="flex justify-between items-center mb-6">
					<h2 className="text-xl font-bold">{selectedBox} Analysis</h2>
					<button
						className="text-gray-500 hover:text-gray-700 transition-colors"
						onClick={onClose}
					>
						<X size={24} /> {/* Lucide React X Icon */}
					</button>
				</div>

				{/* Main Content */}
				<div className="flex border border-black w-full h-[550px]">
					{/* Left Side: Bar Chart (70%) */}
					<div className="pr-4 w-[600px]">
						<Bar
							data={{
								labels: labels,
								datasets: [
									{
										label: "Confidence Score",
										data: values,
										backgroundColor: [
											"rgba(43, 63, 229, 0.8)",
											"rgba(250, 192, 19, 0.8)",
										],
										borderColor: [
											"rgba(43, 63, 229, 1)",
											"rgba(250, 192, 19, 1)",
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
										text: `${selectedBox} Analysis`,
									},
									legend: {
										display: true,
										position: "top",
										labels: {
											boxWidth: 20,
											padding: 10,
										},
									},
								},
								responsive: true,
								maintainAspectRatio: false,
								scales: {
									y: {
										beginAtZero: true,
										max: 100,
										ticks: {
											callback: (value) => `${value}%`,
										},
									},
								},
							}}
						/>
					</div>

					{/* Right Side: Instructions and Response (30%) */}
					<div className=" flex flex-col justify-between border p-4 w-[510px] border-black">
						{/* Instructions Section */}
						<div className="bg-gray-100 p-4 h-[250px] border border-black rounded-lg mb-4">
							<h3 className="text-lg font-semibold mb-2">Instructions</h3>
							<p className="text-sm text-gray-700">
								This is a detailed explanation of how to interpret the{" "}
								{selectedBox.toLowerCase()} analysis.
							</p>
						</div>

						{/* Response Section */}
						<div className="bg-gray-100 p-4 rounded-lg border h-[300px] border-black">
							<h3 className="text-lg font-semibold mb-2">Response</h3>
							<p className="text-sm text-gray-700">
								{selectedBotResponse ? (
									<>
										<div className="flex flex-col w-full p-4 space-y-4">
														{/* User Input on Top */}
														<div className="self-end border border-gray-400 rounded-full px-4 py-2 text-sm bg-gray-100">
															{userInput}
														</div>

														{/* Bot Response Below */}
														<div className="self-start bg-white p-4 rounded-lg border border-gray-400 shadow-md w-64">
															<p className="text-md font-semibold">{selectedBotResponse.text}</p>
															<p className="text-sm text-gray-700 mt-2">
															<strong>Category:</strong> {selectedBotResponse.category} <br />
															<strong>Intent:</strong> {selectedBotResponse.intent} <br />
															<strong>Named Entity Recognition:</strong> {selectedBotResponse.ner}
															</p>
														</div>
											</div>


									</>
								) : (
									"No bot response selected"
								)}
							</p>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
};

export default Modal;
Modal.propTypes = {
	selectedBox: PropTypes.string.isRequired,
	onClose: PropTypes.func.isRequired,
};
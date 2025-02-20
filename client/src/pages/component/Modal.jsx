import PropTypes from "prop-types";
import { BarChart } from "@mui/x-charts/BarChart";
import { X } from "lucide-react"; // Import the X icon from Lucide React

const Modal = ({ selectedBox, onClose }) => {
	// Example data for the bar chart
	const chartData = {
		Intent: { data: [4, 3, 5], labels: ["group A", "group B", "group C"] },
		Category: { data: [1, 6, 3], labels: ["group X", "group Y", "group Z"] },
		NER: { data: [2, 5, 6], labels: ["entity 1", "entity 2", "entity 3"] },
	};

	// Get the data for the selected box
	const { data, labels } = chartData[selectedBox] || {};

	return (
		<div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
			{/* Modal Content */}
			<div className="bg-white rounded-lg max-w-6xl p-6 relative">
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
				<div className="flex border border-black">
					{/* Left Side: Bar Chart (70%) */}
					<div className="pr-4">
						<BarChart
							xAxis={[{ scaleType: "band", data: labels }]}
							series={[{ data }]}
							width={500}
							height={300}
						/>
					</div>

					{/* Right Side: Instructions and Response (30%) */}
					<div className=" flex flex-col justify-between border border-black">
						{/* Instructions Section */}
						<div className="bg-gray-100 p-4 rounded-lg mb-4">
							<h3 className="text-lg font-semibold mb-2">Instructions</h3>
							<p className="text-sm text-gray-700">
								This is a detailed explanation of how to interpret the{" "}
								{selectedBox.toLowerCase()} analysis.
							</p>
						</div>

						{/* Response Section */}
						<div className="bg-gray-100 p-4 rounded-lg border border-black">
							<h3 className="text-lg font-semibold mb-2">Response</h3>
							<p className="text-sm text-gray-700">
								The response based on the {selectedBox.toLowerCase()} analysis
								will appear here.
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

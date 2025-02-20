import PropTypes from "prop-types";
import { useState } from "react";
import BarChart from "./component/BarChart"; // Import the reusable Box component
import Modal from "./component/Modal"; // Import the Modal component

const Chat = () => {
	// State to manage modal visibility and selected box
	const [isModalOpen, setIsModalOpen] = useState(false);
	const [selectedBox, setSelectedBox] = useState(null);

	// Handlers for each box click
	const handleBoxClick = (boxType) => {
		setSelectedBox(boxType); // Set the selected box type
		setIsModalOpen(true); // Open the modal
	};

	return (
		<div className="p-6 ">
			{/* Main Title */}
			<h1 className="text-3xl font-bold text-center mb-8">Chat Analysis</h1>

			{/* User Message Display Section */}
			<div className="mb-8 p-4 bg-white border border-gray-200 rounded-lg shadow-md">
				<p className="text-lg text-gray-700">Displaying user message here:</p>
			</div>

			{/* Container for the grid */}
			<div className="grid grid-cols-3 gap-6">
				{/* Intent Box */}
				<BarChart
					title="Intent"
					description="Analyze user intents"
					color="#3B82F6" // Blue
					onClick={() => handleBoxClick("Intent")}
				/>
				{/* Category Box */}
				<BarChart
					title="Category"
					description="Classify message categories"
					color="#10B981" // Green
					onClick={() => handleBoxClick("Category")}
				/>
				{/* NER Box */}
				<BarChart
					title="NER"
					description="Named Entity Recognition"
					color="#EF4444" // Red
					onClick={() => handleBoxClick("NER")}
				/>
			</div>

			{/* Modal */}
			{isModalOpen && (
				<Modal
					selectedBox={selectedBox}
					onClose={() => setIsModalOpen(false)} // Close modal handler
				/>
			)}
		</div>
	);
};

export default Chat;
Chat.propTypes = {
	selectedBox: PropTypes.string,
	onClose: PropTypes.func,
};
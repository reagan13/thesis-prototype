import PropTypes from "prop-types";
import { Send } from "lucide-react";
const InputSection = ({ input, setInput, handleSend }) => (
	<div className="flex justify-between items-center gap-4">
		<input
			type="text"
			value={input}
			onChange={(e) => setInput(e.target.value)}
			placeholder="Enter your message..."
			className="rounded-full w-full border border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300 py-4 px-5"
		/>
		<button
			onClick={handleSend}
			className="bg-blue-500 text-darkNavy rounded-full p-2 hover:bg-blue-600 transition duration-200"
		>
			<Send />
		</button>
	</div>
);
InputSection.propTypes = {
	input: PropTypes.string,
	setInput: PropTypes.func,
	handleSend: PropTypes.func,
};

export default InputSection;

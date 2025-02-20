import PropTypes from "prop-types";
import { Send } from "lucide-react";

const InputSection = ({ input, setInput, handleSend }) => (
	<div className="flex items-center bg-transparent">
		<div className="flex items-center w-full border-2 border-[#111852] rounded-2xl overflow-hidden bg-white focus-within:ring-2 focus-within:ring-[#111852] outline outline-2 outline-white outline-offset-2 ">
			<input
				type="text"
				value={input}
				onChange={(e) => setInput(e.target.value)}
				placeholder="Type a new message here"
				className="w-full px-4 py-3 text-gray-700 focus:outline-none"
			/>
			<button
				onClick={handleSend}
				className="p-3 text-[#111852] hover:text-blue-600 transition duration-200"
			>
				<Send size={20} />
			</button>
		</div>
	</div>
);

InputSection.propTypes = {
	input: PropTypes.string,
	setInput: PropTypes.func,
	handleSend: PropTypes.func,
};

export default InputSection;

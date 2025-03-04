import PropTypes from "prop-types";

const UserMessage = ({ text, timestamp }) => (
	<div className="flex items-end gap-2">
		{/* Message Bubble with Timestamp */}
		<div className="flex flex-col items-end">
			<div
				className={`relative p-3 max-w-[500px] text-left bg-white border border-black ${
					text.length > 100
						? "rounded-t-3xl rounded-l-3x rounded-br-none rounded-tr-3xl"
						: "rounded-t-full rounded-l-full rounded-br-none rounded-tr-full"
				}`}
			>
				<p className="whitespace-pre-wrap">{text}</p>
			</div>
			{/* Timestamp */}
			<p className="text-[13px] text-gray-500 mt-1">{timestamp}</p>
		</div>

		{/* User Avatar (on the right side) */}
		<img
			src={"../../public/user.png"}
			alt="User Avatar"
			className="w-14 h-14 rounded-full"
		/>
	</div>
);

UserMessage.propTypes = {
	text: PropTypes.string.isRequired,
	timestamp: PropTypes.string.isRequired,
};

export default UserMessage;

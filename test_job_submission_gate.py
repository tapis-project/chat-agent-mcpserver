"""Submission gate regressions; no model calls, credentials, or remote jobs."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from mcp.types import CallToolResult, TextContent
from response_schemas import AgentResponse, ExecutionPlan, ToolObservation
from tool_observation import observation_interceptor


def load_client():
    path = Path(__file__).with_name('tapis_mcp_client_header.py')
    spec = importlib.util.spec_from_file_location('submission_gate_test_client', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'app_config': SimpleNamespace(config={'llm_provider':'openai','llm_name':'test-model'})}), patch('dotenv.load_dotenv'), patch('langchain_openai.ChatOpenAI'):
        spec.loader.exec_module(module)
    return module


client = load_client()


class SubmissionGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.trace = []
        self.args = dict(name='HelloWorld', description='test', app_id='hello-world-tapis-test-0.0.1', app_version='0.0.1', exec_system_id='S3-test-ajamthe')
        self.good_app = {'status':'success','result':{'id':self.args['app_id'],'version':self.args['app_version']}}
        self.good_system = {'status':'success','result':{'id':self.args['exec_system_id']}}

    def tool(self, name, data=None, *, ok=True, raises=False, missing_trace=False):
        async def invoke(arguments):
            if raises:
                raise ValueError('argument validation failed')
            if missing_trace:
                return []
            observation = ToolObservation(ok=ok, tool=name, data=data, error_code=None if ok else 'http_error', message=None if ok else 'Upstream returned HTTP 404')
            raw = CallToolResult(content=[TextContent(type='text',text=observation.model_dump_json())])
            result = await observation_interceptor(self.trace)(SimpleNamespace(name=name), AsyncMock(return_value=raw))
            return [{'type':'text','text':result.content[0].text}]
        return SimpleNamespace(name=name, description='', args_schema={}, ainvoke=AsyncMock(side_effect=invoke))

    async def chat(self, app, system=None, args=None, calls=None, final_response=None):
        submit = self.tool('submit_job', {'job_uuid':'job-test-123'})
        poll = self.tool('get_job_status', {'job_uuid':'job-test-123','status':'PENDING'})
        tools = [t for t in (app,system,submit,poll) if t is not None]
        # A preceding system list passes the old static sequence check but cannot
        # establish that the selected application exists.
        listing = self.tool('get_systems_list', {'status':'success','result':[]})
        tools.append(listing)
        plan = ExecutionPlan(thought_process='Test missing application gate', proposed_calls=calls if calls is not None else [{'tool':'get_systems_list'}, {'tool':'submit_job','arguments':self.args if args is None else args}])
        final = final_response or AgentResponse(status='completed',answer='Job submitted',steps=[],total_count=None,citations=[],completed_actions=['submit_job'],missing_fields=[])
        planner = SimpleNamespace(ainvoke=AsyncMock(return_value=plan))
        summarizer = SimpleNamespace(ainvoke=AsyncMock(return_value=final))
        model = SimpleNamespace(with_structured_output=Mock(side_effect=lambda schema, **kwargs: planner if schema is ExecutionPlan else summarizer))
        async def get_client(token, trace):
            self.trace = trace
            return SimpleNamespace(get_tools=AsyncMock(return_value=tools))
        with patch.object(client,'basic_model',model), patch.object(client,'get_client',get_client), patch.object(client.asyncio,'sleep',AsyncMock()):
            result = await client.chat(SimpleNamespace(headers={'x-tapis-token':'test'}), SimpleNamespace(message='Submit HelloWorld'))
        self.assertEqual([e.step for e in result.trace],list(range(1,len(result.trace)+1)))
        self.assertEqual(result.execution_plan, plan)
        return result, submit, poll, summarizer

    async def assert_blocked(self, app, system=None, args=None):
        result, submit, poll, summary = await self.chat(app,system,args)
        self.assertEqual(result.status,'blocked')
        self.assertEqual(result.trace[-1].observation.error_code,'PRECONDITION_FAILED')
        self.assertEqual(result.completed_actions,[])
        submit.ainvoke.assert_not_awaited()
        poll.ainvoke.assert_not_awaited()
        summary.ainvoke.assert_not_awaited()
        return result

    async def test_missing_app_blocks_submission_even_when_system_exists(self):
        app = self.tool('get_app_info',ok=False)
        system = self.tool('get_system',self.good_system)
        result = await self.assert_blocked(app,system)
        system.ainvoke.assert_not_awaited()
        app.ainvoke.assert_awaited_once_with({'appId':self.args['app_id'],'appVersion':'0.0.1'})
        self.assertFalse(result.trace[1].observation.ok)

    async def test_missing_system_blocks_submission(self):
        await self.assert_blocked(self.tool('get_app_info',self.good_app),self.tool('get_system',ok=False))

    async def test_missing_app_tool_blocks_submission(self):
        await self.assert_blocked(None,self.tool('get_system',self.good_system))

    async def test_missing_system_tool_blocks_submission(self):
        await self.assert_blocked(self.tool('get_app_info',self.good_app))

    async def test_malformed_and_mismatched_app_results_block(self):
        for data in (None, {'result':[]}, {'result':{}}, {'result':{'id':'other','version':'0.0.1'}}, {'result':{'id':self.args['app_id'],'version':'different'}}, {'status':'error','result':self.good_app['result']}):
            with self.subTest(data=data):
                await self.assert_blocked(self.tool('get_app_info',data),self.tool('get_system',self.good_system))

    async def test_mismatched_system_blocks(self):
        await self.assert_blocked(self.tool('get_app_info',self.good_app),self.tool('get_system',{'result':{'id':'another-system'}}))

    async def test_missing_required_fields_block(self):
        for field in ('app_id','app_version','exec_system_id'):
            with self.subTest(field=field):
                args = {k:v for k,v in self.args.items() if k != field}
                app = self.tool('get_app_info',self.good_app)
                result = await self.assert_blocked(app,self.tool('get_system',self.good_system),args)
                self.assertIn(field,result.answer)
                app.ainvoke.assert_not_awaited()

    async def test_precheck_exception_blocks(self):
        await self.assert_blocked(self.tool('get_app_info',raises=True),self.tool('get_system',self.good_system))

    async def test_unrecorded_precheck_aborts_without_submission(self):
        # No normalized observation is evidence of a protocol failure, not access.
        submit = self.tool('submit_job',{'job_uuid':'job-test'})
        tools = {'get_app_info':self.tool('get_app_info',missing_trace=True),'submit_job':submit}
        with self.assertRaises(client.HTTPException):
            await client.validate_job_preconditions(self.args,tools,self.trace)
        submit.ainvoke.assert_not_awaited()

    async def test_valid_app_and_system_allow_submission_and_poll(self):
        app = self.tool('get_app_info',self.good_app)
        system = self.tool('get_system',self.good_system)
        result, submit, poll, _ = await self.chat(app,system)
        self.assertEqual(result.status,'completed')
        self.assertEqual([e.tool for e in result.trace],['get_systems_list','get_app_info','get_system','submit_job','get_job_status'])
        system.ainvoke.assert_awaited_once_with({'systemId':self.args['exec_system_id']})
        submit.ainvoke.assert_awaited_once_with(self.args)
        poll.ainvoke.assert_awaited_once_with({'job_uuid':'job-test-123'})
        self.assertTrue(result.trace[3].mutating)

    async def test_static_denial_returns_plan_without_executing_tools(self):
        app = self.tool('get_app_info', self.good_app)
        result, submit, poll, summary = await self.chat(app, calls=[{'tool':'delete_system'}])
        self.assertEqual(result.status, 'blocked')
        self.assertIn('prohibited', result.plan_validation_error)
        self.assertEqual(result.trace, [])
        app.ainvoke.assert_not_awaited()
        submit.ainvoke.assert_not_awaited()
        summary.ainvoke.assert_not_awaited()

    async def test_read_failure_is_preserved_without_duplicate_trace(self):
        failed = AgentResponse(status='failed', answer='Lookup failed', steps=[], total_count=None, citations=[], completed_actions=[], missing_fields=[])
        app = self.tool('get_app_info', ok=False)
        result, submit, _, _ = await self.chat(app, calls=[{'tool':'get_app_info', 'arguments':{'appId':'missing','appVersion':'0.0.1'}}], final_response=failed)
        self.assertEqual(len(result.trace), 1)
        self.assertFalse(result.trace[0].observation.ok)
        self.assertEqual(result.trace[0].observation.error_code, 'http_error')
        submit.ainvoke.assert_not_awaited()

    async def test_forged_completed_action_is_rejected(self):
        with self.assertRaises(client.HTTPException) as exc:
            await self.chat(None, calls=[{'tool':'get_systems_list'}])
        self.assertEqual(exc.exception.status_code, 502)
        self.assertIn('without successful tool evidence', exc.exception.detail)

    def test_root_does_not_echo_credentials(self):
        result = client.root(SimpleNamespace(headers={'x-tapis-token':'do-not-echo'}))
        self.assertNotIn('do-not-echo', str(result))


if __name__ == '__main__':
    unittest.main()
